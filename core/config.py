"""Configuration management for the RAG Agent system.

Provides centralized configuration with environment variable support,
validation, and type-safe access to settings.
"""
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from core.exceptions import ConfigurationException, MissingConfigException, InvalidConfigException
from core.validation import ConfigValidator


class RetrievalMode(str, Enum):
    """Supported retrieval modes."""
    PARRAG = "parrag"
    HOPRAG = "hoprag"
    MODULAR = "modular"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelConfig:
    """Configuration for language models."""
    
    # Model selection
    use_ollama: bool = False
    ollama_model: str = "llama2"
    ollama_embed_model: str = "qwen3-embedding:latest"
    
    # Model parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    
    # Advanced features
    enable_streaming: bool = False
    enable_function_calling: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        ConfigValidator.validate_float_range(
            self.temperature,
            "temperature",
            min_val=0.0,
            max_val=2.0
        )
        ConfigValidator.validate_positive_int(self.max_tokens, "max_tokens")
        ConfigValidator.validate_positive_int(self.timeout, "timeout")


@dataclass
class RetrievalConfig:
    """Configuration for retrieval systems."""
    
    # Retrieval parameters
    top_k: int = 10
    similarity_threshold: float = 0.7
    min_relevance_score: float = 0.6
    
    # Index settings
    enable_faiss: bool = False
    faiss_index_path: str = "data/faiss_index.iv"
    faiss_mapping_path: str = "data/faiss_mapping.json"
    
    # Hybrid search
    enable_hybrid: bool = False
    hybrid_alpha: float = 0.5  # Weight for vector vs BM25
    
    # Reranking
    enable_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        ConfigValidator.validate_positive_int(self.top_k, "top_k")
        ConfigValidator.validate_float_range(
            self.similarity_threshold,
            "similarity_threshold"
        )
        ConfigValidator.validate_float_range(
            self.min_relevance_score,
            "min_relevance_score"
        )
        ConfigValidator.validate_float_range(
            self.hybrid_alpha,
            "hybrid_alpha"
        )


@dataclass
class VerificationConfig:
    """Configuration for answer verification."""
    
    # Verification thresholds
    min_support_score: float = 0.7
    hallucination_threshold: float = 0.3
    max_retries: int = 2
    
    # Self-RAG
    enable_self_rag: bool = False
    enable_correction: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        ConfigValidator.validate_float_range(
            self.min_support_score,
            "min_support_score"
        )
        ConfigValidator.validate_float_range(
            self.hallucination_threshold,
            "hallucination_threshold"
        )
        ConfigValidator.validate_positive_int(self.max_retries, "max_retries")


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    
    # Memory settings
    enable_memory: bool = False
    max_turns: int = 10
    max_tokens: int = 4000
    storage_path: str = "data/memory"
    
    # Session management
    session_timeout: int = 3600  # 1 hour
    enable_user_profiles: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        ConfigValidator.validate_positive_int(self.max_turns, "max_turns")
        ConfigValidator.validate_positive_int(self.max_tokens, "max_tokens")
        ConfigValidator.validate_positive_int(self.session_timeout, "session_timeout")


@dataclass
class CachingConfig:
    """Configuration for caching."""
    
    # Cache settings
    enable_query_cache: bool = True
    query_cache_ttl: int = 300  # 5 minutes
    query_cache_size: int = 1000
    
    enable_embedding_cache: bool = True
    embedding_cache_size: int = 10000
    
    enable_retrieval_cache: bool = True
    retrieval_cache_ttl: int = 600  # 10 minutes
    retrieval_cache_size: int = 500
    
    def __post_init__(self):
        """Validate configuration."""
        ConfigValidator.validate_positive_int(self.query_cache_ttl, "query_cache_ttl")
        ConfigValidator.validate_positive_int(self.query_cache_size, "query_cache_size")
        ConfigValidator.validate_positive_int(
            self.embedding_cache_size,
            "embedding_cache_size"
        )
        ConfigValidator.validate_positive_int(
            self.retrieval_cache_ttl,
            "retrieval_cache_ttl"
        )
        ConfigValidator.validate_positive_int(
            self.retrieval_cache_size,
            "retrieval_cache_size"
        )


@dataclass
class APIConfig:
    """Configuration for API server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Security
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_rate_limiting: bool = True
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10
    
    # Features
    enable_compression: bool = True
    enable_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        ConfigValidator.validate_positive_int(self.port, "port")
        ConfigValidator.validate_positive_int(self.workers, "workers")
        ConfigValidator.validate_positive_int(self.rate_limit_rpm, "rate_limit_rpm")
        ConfigValidator.validate_positive_int(self.rate_limit_burst, "rate_limit_burst")


@dataclass
class ObservabilityConfig:
    """Configuration for observability."""
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Optional[str] = None
    
    # Metrics
    enable_prometheus: bool = True
    metrics_port: int = 9090
    
    # Tracing
    enable_tracing: bool = False
    trace_sampling_rate: float = 0.1
    
    def __post_init__(self):
        """Validate configuration."""
        if self.log_file:
            ConfigValidator.validate_path(self.log_file, "log_file", must_exist=False)
        ConfigValidator.validate_positive_int(self.metrics_port, "metrics_port")
        ConfigValidator.validate_float_range(
            self.trace_sampling_rate,
            "trace_sampling_rate"
        )


@dataclass
class RAGConfig:
    """Main configuration for RAG Agent system."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    # Global settings
    data_dir: str = "data"
    enable_advanced_features: bool = False
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables.
        
        Returns:
            RAGConfig instance populated from environment
        """
        config = cls()
        
        # Model configuration
        config.model.use_ollama = os.getenv("USE_OLLAMA", "0") == "1"
        config.model.ollama_model = os.getenv("OLLAMA_MODEL", config.model.ollama_model)
        config.model.ollama_embed_model = os.getenv(
            "OLLAMA_EMBED_MODEL",
            config.model.ollama_embed_model
        )
        config.model.timeout = int(os.getenv("MODEL_TIMEOUT", str(config.model.timeout)))
        
        # Retrieval configuration
        config.retrieval.enable_faiss = os.getenv("ENABLE_FAISS", "0") == "1"
        config.retrieval.enable_hybrid = os.getenv("ENABLE_HYBRID", "0") == "1"
        config.retrieval.enable_reranking = os.getenv("ENABLE_RERANKING", "0") == "1"
        config.retrieval.top_k = int(os.getenv("RETRIEVAL_TOP_K", str(config.retrieval.top_k)))
        
        # Verification configuration
        config.verification.enable_self_rag = os.getenv("ENABLE_SELF_RAG", "0") == "1"
        config.verification.max_retries = int(
            os.getenv("MAX_RETRIES", str(config.verification.max_retries))
        )
        
        # Memory configuration
        config.memory.enable_memory = os.getenv("ENABLE_MEMORY", "0") == "1"
        
        # API configuration
        config.api.port = int(os.getenv("PORT", str(config.api.port)))
        config.api.enable_rate_limiting = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
        config.api.rate_limit_rpm = int(
            os.getenv("RATE_LIMIT_RPM", str(config.api.rate_limit_rpm))
        )
        cors_origins = os.getenv("CORS_ORIGINS", "*")
        config.api.cors_origins = [o.strip() for o in cors_origins.split(",")]
        
        # Observability configuration
        log_level_str = os.getenv("LOG_LEVEL", "INFO")
        try:
            config.observability.log_level = LogLevel[log_level_str.upper()]
        except KeyError:
            config.observability.log_level = LogLevel.INFO
        
        # Global settings
        config.enable_advanced_features = os.getenv("USE_ADVANCED_RAG", "0") == "1"
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        from dataclasses import asdict
        return asdict(self)
    
    def validate(self) -> None:
        """Validate entire configuration.
        
        Raises:
            ConfigurationException: If configuration is invalid
        """
        # Data directory must exist or be creatable
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir, exist_ok=True)
            except Exception as e:
                raise ConfigurationException(
                    f"Cannot create data directory: {self.data_dir}",
                    details={"error": str(e)}
                )
        
        # If FAISS is enabled, check index paths
        if self.retrieval.enable_faiss:
            index_dir = os.path.dirname(self.retrieval.faiss_index_path)
            if not os.path.exists(index_dir):
                try:
                    os.makedirs(index_dir, exist_ok=True)
                except Exception as e:
                    raise ConfigurationException(
                        f"Cannot create FAISS index directory: {index_dir}",
                        details={"error": str(e)}
                    )
        
        # Memory storage path
        if self.memory.enable_memory:
            if not os.path.exists(self.memory.storage_path):
                try:
                    os.makedirs(self.memory.storage_path, exist_ok=True)
                except Exception as e:
                    raise ConfigurationException(
                        f"Cannot create memory storage directory: {self.memory.storage_path}",
                        details={"error": str(e)}
                    )


# Global configuration instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get global configuration instance.
    
    Returns:
        RAGConfig instance
    """
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
        _config.validate()
    return _config


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None
