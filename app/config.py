"""Configuration management for the RAG system."""
import os
from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "ollama"
    model_name: str = "llama2"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 30


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    provider: str = "ollama"
    model_name: str = "qwen3-embedding:latest"
    dimension: int = 768
    batch_size: int = 32


class RerankerConfig(BaseModel):
    """Reranker configuration."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    max_length: int = 512


class ModelsConfig(BaseModel):
    """Models configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class FeatureFlags(BaseModel):
    """Feature flags for enabling/disabling capabilities."""
    multi_agent_mode: bool = False
    graphrag_enabled: bool = True
    self_rag_verification: bool = True
    intent_routing: bool = True
    hybrid_retrieval: bool = True
    cross_encoder_reranking: bool = True
    hierarchical_memory: bool = True
    human_in_the_loop: bool = False


class IntentClassifierConfig(BaseModel):
    """Intent classifier configuration."""
    enabled: bool = True
    confidence_threshold: float = 0.75
    smalltalk_patterns: list = Field(default_factory=lambda: [
        "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank you"
    ])


class FAQMatcherConfig(BaseModel):
    """FAQ matcher configuration."""
    enabled: bool = True
    similarity_threshold: float = 0.85
    max_results: int = 3


class SafetyConfig(BaseModel):
    """Safety configuration."""
    enabled: bool = True
    jailbreak_detection: bool = True
    max_query_length: int = 1000


class RouterConfig(BaseModel):
    """Router configuration."""
    intent_classifier: IntentClassifierConfig = Field(default_factory=IntentClassifierConfig)
    faq_matcher: FAQMatcherConfig = Field(default_factory=FAQMatcherConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class HybridRetrievalConfig(BaseModel):
    """Hybrid retrieval configuration."""
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    rrf_k: int = 60
    candidate_pool_size: int = 50


class RerankingConfig(BaseModel):
    """Reranking configuration."""
    enabled: bool = True
    top_k: int = 10
    min_score: float = 0.0


class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    strategy: str = "semantic"
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class Neo4jConfig(BaseModel):
    """Neo4j configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class EntityExtractionConfig(BaseModel):
    """Entity extraction configuration."""
    enabled: bool = True
    entity_types: list = Field(default_factory=lambda: [
        "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"
    ])
    relation_types: list = Field(default_factory=lambda: [
        "RELATED_TO", "WORKS_AT", "LOCATED_IN", "DEPENDS_ON"
    ])


class TraversalConfig(BaseModel):
    """Graph traversal configuration."""
    max_hops: int = 3
    max_nodes: int = 100
    min_relevance: float = 0.5


class GraphRAGConfig(BaseModel):
    """GraphRAG configuration."""
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    entity_extraction: EntityExtractionConfig = Field(default_factory=EntityExtractionConfig)
    traversal: TraversalConfig = Field(default_factory=TraversalConfig)


class RetrievalGradingConfig(BaseModel):
    """Retrieval grading configuration."""
    enabled: bool = True
    min_relevance_score: float = 0.6
    max_retries: int = 2


class AnswerVerificationConfig(BaseModel):
    """Answer verification configuration."""
    enabled: bool = True
    min_support_score: float = 0.7
    hallucination_threshold: float = 0.3


class CorrectionPolicyConfig(BaseModel):
    """Correction policy configuration."""
    re_retrieve_threshold: float = 0.5
    refuse_threshold: float = 0.3
    hedge_low_confidence: bool = True


class VerificationConfig(BaseModel):
    """Verification configuration."""
    enabled: bool = True
    retrieval_grading: RetrievalGradingConfig = Field(default_factory=RetrievalGradingConfig)
    answer_verification: AnswerVerificationConfig = Field(default_factory=AnswerVerificationConfig)
    correction_policy: CorrectionPolicyConfig = Field(default_factory=CorrectionPolicyConfig)


class ShortTermMemoryConfig(BaseModel):
    """Short-term memory configuration."""
    enabled: bool = True
    max_turns: int = 5
    max_tokens: int = 2000


class SessionMemoryConfig(BaseModel):
    """Session memory configuration."""
    enabled: bool = True
    max_turns: int = 20
    max_tokens: int = 8000
    summarize_after: int = 10


class LongTermMemoryConfig(BaseModel):
    """Long-term memory configuration."""
    enabled: bool = True
    max_facts: int = 100
    relevance_threshold: float = 0.6


class SemanticSwitchingConfig(BaseModel):
    """Semantic switching configuration."""
    enabled: bool = True
    topic_similarity_threshold: float = 0.7


class MemoryConfig(BaseModel):
    """Memory configuration."""
    short_term: ShortTermMemoryConfig = Field(default_factory=ShortTermMemoryConfig)
    session: SessionMemoryConfig = Field(default_factory=SessionMemoryConfig)
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)
    semantic_switching: SemanticSwitchingConfig = Field(default_factory=SemanticSwitchingConfig)


class OrchestrationConfig(BaseModel):
    """Orchestration configuration."""
    max_iterations: int = 10
    timeout_seconds: int = 60
    checkpoint_enabled: bool = True
    checkpoint_backend: str = "memory"
    human_approval: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "confidence_threshold": 0.5,
        "timeout_seconds": 300
    })


class AgentRoleConfig(BaseModel):
    """Agent role configuration."""
    enabled: bool = True
    model: str = "llama2"


class AgentsConfig(BaseModel):
    """Multi-agent configuration."""
    framework: str = "autogen"
    roles: Dict[str, AgentRoleConfig] = Field(default_factory=lambda: {
        "planner": AgentRoleConfig(),
        "extractor": AgentRoleConfig(),
        "qa_specialist": AgentRoleConfig(),
        "judge": AgentRoleConfig(),
        "finalizer": AgentRoleConfig()
    })


class TracingConfig(BaseModel):
    """Tracing configuration."""
    enabled: bool = True
    backend: str = "langsmith"
    sample_rate: float = 1.0


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    enabled: bool = True
    export_prometheus: bool = True
    export_interval_seconds: int = 60


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    redact_pii: bool = True


class ArtifactsConfig(BaseModel):
    """Artifacts configuration."""
    save_retrieved_passages: bool = True
    save_prompts: bool = True
    save_intermediate_steps: bool = True
    retention_days: int = 30


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 60
    retrieval_timeout_seconds: int = 10
    generation_timeout_seconds: int = 30
    caching: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "ttl_seconds": 3600,
        "max_size": 1000
    })


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    enabled: bool = True
    test_suite_path: str = "tests/scenario/"
    metrics: list = Field(default_factory=lambda: [
        "accuracy", "hallucination_rate", "latency_p50",
        "latency_p95", "cost_per_query", "fallback_rate"
    ])
    seeds: list = Field(default_factory=lambda: [42, 123, 456])


class Config(BaseModel):
    """Main configuration class."""
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graphrag: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    class Config:
        arbitrary_types_allowed = True


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file or environment variables.
    
    Args:
        config_path: Path to YAML config file. Defaults to configs/default.yaml
    
    Returns:
        Config object
    """
    if config_path is None:
        config_path = os.getenv(
            "RAG_CONFIG_PATH",
            str(Path(__file__).parent.parent / "configs" / "default.yaml")
        )
    
    # Load YAML config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
    
    # Override with environment variables
    config_dict = _apply_env_overrides(config_dict)
    
    return Config(**config_dict)


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config.
    
    Environment variables use the format: RAG_SECTION_SUBSECTION_KEY
    Example: RAG_FEATURES_GRAPHRAG_ENABLED=true
    """
    env_prefix = "RAG_"
    
    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue
        
        # Parse the key path
        path_parts = key[len(env_prefix):].lower().split('_')
        
        # Navigate and set the value
        current = config_dict
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Convert string value to appropriate type
        final_key = path_parts[-1]
        if value.lower() in ('true', 'false'):
            current[final_key] = value.lower() == 'true'
        elif value.isdigit():
            current[final_key] = int(value)
        elif value.replace('.', '', 1).isdigit():
            current[final_key] = float(value)
        else:
            current[final_key] = value
    
    return config_dict


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_path)
    return _config
