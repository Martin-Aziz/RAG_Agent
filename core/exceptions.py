"""Custom exceptions for the RAG Agent system.

This module defines a hierarchy of custom exceptions for better error handling
and more informative error messages throughout the application.
"""
from typing import Optional, Dict, Any


class RAGAgentException(Exception):
    """Base exception for all RAG Agent errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


# Retrieval Exceptions
class RetrievalException(RAGAgentException):
    """Base exception for retrieval errors."""
    pass


class EmptyRetrievalException(RetrievalException):
    """Raised when retrieval returns no results."""
    
    def __init__(self, query: str, retriever_type: str = "unknown"):
        super().__init__(
            f"No documents retrieved for query: '{query[:100]}'",
            error_code="EMPTY_RETRIEVAL",
            details={"query": query, "retriever_type": retriever_type}
        )


class RetrievalTimeoutException(RetrievalException):
    """Raised when retrieval operation times out."""
    
    def __init__(self, timeout: float, retriever_type: str = "unknown"):
        super().__init__(
            f"Retrieval timed out after {timeout}s",
            error_code="RETRIEVAL_TIMEOUT",
            details={"timeout": timeout, "retriever_type": retriever_type}
        )


class IndexNotFoundException(RetrievalException):
    """Raised when index file is not found."""
    
    def __init__(self, index_path: str):
        super().__init__(
            f"Index not found at: {index_path}",
            error_code="INDEX_NOT_FOUND",
            details={"index_path": index_path}
        )


# Generation Exceptions
class GenerationException(RAGAgentException):
    """Base exception for generation errors."""
    pass


class ModelException(GenerationException):
    """Raised when model fails to generate."""
    
    def __init__(self, model_name: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Model '{model_name}' failed to generate response",
            error_code="MODEL_FAILURE",
            details={
                "model_name": model_name,
                "original_error": str(original_error) if original_error else None
            }
        )


class PromptTooLongException(GenerationException):
    """Raised when prompt exceeds model's context window."""
    
    def __init__(self, prompt_length: int, max_length: int):
        super().__init__(
            f"Prompt length ({prompt_length}) exceeds maximum ({max_length})",
            error_code="PROMPT_TOO_LONG",
            details={"prompt_length": prompt_length, "max_length": max_length}
        )


class ResponseParsingException(GenerationException):
    """Raised when model response cannot be parsed."""
    
    def __init__(self, response: str, expected_format: str = "JSON"):
        super().__init__(
            f"Failed to parse {expected_format} from model response",
            error_code="RESPONSE_PARSING_FAILED",
            details={"response_preview": response[:200], "expected_format": expected_format}
        )


# Verification Exceptions
class VerificationException(RAGAgentException):
    """Base exception for verification errors."""
    pass


class HallucinationDetectedException(VerificationException):
    """Raised when hallucination is detected in generated answer."""
    
    def __init__(self, confidence: float, contradictions: list):
        super().__init__(
            "Hallucination detected in generated answer",
            error_code="HALLUCINATION_DETECTED",
            details={
                "confidence": confidence,
                "contradictions": contradictions
            }
        )


class EvidenceMismatchException(VerificationException):
    """Raised when answer doesn't match provided evidence."""
    
    def __init__(self, unsupported_claims: list):
        super().__init__(
            "Answer contains unsupported claims",
            error_code="EVIDENCE_MISMATCH",
            details={"unsupported_claims": unsupported_claims}
        )


# Orchestration Exceptions
class OrchestrationException(RAGAgentException):
    """Base exception for orchestration errors."""
    pass


class MaxRetriesExceededException(OrchestrationException):
    """Raised when maximum retry attempts are exceeded."""
    
    def __init__(self, max_retries: int, operation: str):
        super().__init__(
            f"Maximum retries ({max_retries}) exceeded for operation: {operation}",
            error_code="MAX_RETRIES_EXCEEDED",
            details={"max_retries": max_retries, "operation": operation}
        )


class InvalidModeException(OrchestrationException):
    """Raised when an invalid orchestration mode is specified."""
    
    def __init__(self, mode: str, valid_modes: list):
        super().__init__(
            f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}",
            error_code="INVALID_MODE",
            details={"mode": mode, "valid_modes": valid_modes}
        )


# Configuration Exceptions
class ConfigurationException(RAGAgentException):
    """Base exception for configuration errors."""
    pass


class MissingConfigException(ConfigurationException):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str):
        super().__init__(
            f"Missing required configuration: {config_key}",
            error_code="MISSING_CONFIG",
            details={"config_key": config_key}
        )


class InvalidConfigException(ConfigurationException):
    """Raised when configuration value is invalid."""
    
    def __init__(self, config_key: str, value: Any, reason: str):
        super().__init__(
            f"Invalid configuration for '{config_key}': {reason}",
            error_code="INVALID_CONFIG",
            details={"config_key": config_key, "value": str(value), "reason": reason}
        )


# Memory Exceptions
class MemoryException(RAGAgentException):
    """Base exception for memory errors."""
    pass


class SessionNotFoundException(MemoryException):
    """Raised when session is not found."""
    
    def __init__(self, session_id: str):
        super().__init__(
            f"Session not found: {session_id}",
            error_code="SESSION_NOT_FOUND",
            details={"session_id": session_id}
        )


class MemoryCapacityException(MemoryException):
    """Raised when memory capacity is exceeded."""
    
    def __init__(self, current_size: int, max_size: int):
        super().__init__(
            f"Memory capacity exceeded: {current_size}/{max_size}",
            error_code="MEMORY_CAPACITY_EXCEEDED",
            details={"current_size": current_size, "max_size": max_size}
        )


# Input Validation Exceptions
class ValidationException(RAGAgentException):
    """Base exception for input validation errors."""
    pass


class InvalidQueryException(ValidationException):
    """Raised when query is invalid."""
    
    def __init__(self, query: str, reason: str):
        super().__init__(
            f"Invalid query: {reason}",
            error_code="INVALID_QUERY",
            details={"query": query[:100], "reason": reason}
        )


class QueryTooLongException(ValidationException):
    """Raised when query exceeds maximum length."""
    
    def __init__(self, query_length: int, max_length: int):
        super().__init__(
            f"Query too long: {query_length} characters (max: {max_length})",
            error_code="QUERY_TOO_LONG",
            details={"query_length": query_length, "max_length": max_length}
        )


# Resource Exceptions
class ResourceException(RAGAgentException):
    """Base exception for resource errors."""
    pass


class ResourceNotFoundException(ResourceException):
    """Raised when a required resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            error_code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ResourceExhaustedException(ResourceException):
    """Raised when system resources are exhausted."""
    
    def __init__(self, resource_type: str, limit: Optional[int] = None):
        super().__init__(
            f"{resource_type} exhausted" + (f" (limit: {limit})" if limit else ""),
            error_code="RESOURCE_EXHAUSTED",
            details={"resource_type": resource_type, "limit": limit}
        )


# External Service Exceptions
class ExternalServiceException(RAGAgentException):
    """Base exception for external service errors."""
    pass


class ModelUnavailableException(ExternalServiceException):
    """Raised when external model service is unavailable."""
    
    def __init__(self, model_name: str, service: str):
        super().__init__(
            f"Model '{model_name}' unavailable on service '{service}'",
            error_code="MODEL_UNAVAILABLE",
            details={"model_name": model_name, "service": service}
        )


class ServiceTimeoutException(ExternalServiceException):
    """Raised when external service times out."""
    
    def __init__(self, service: str, timeout: float):
        super().__init__(
            f"Service '{service}' timed out after {timeout}s",
            error_code="SERVICE_TIMEOUT",
            details={"service": service, "timeout": timeout}
        )
