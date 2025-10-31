"""Input validation utilities for the RAG Agent system.

This module provides comprehensive validation for user inputs, queries,
and configuration to ensure data integrity and security.
"""
import re
from typing import Optional, List, Dict, Any, Tuple
from core.exceptions import (
    InvalidQueryException,
    QueryTooLongException,
    ValidationException
)


# Security patterns
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # XSS attempts
    r'javascript:',  # JavaScript protocol
    r'on\w+\s*=',  # Event handlers
    r'eval\s*\(',  # eval() calls
    r'exec\s*\(',  # exec() calls
    r'__import__',  # Python imports
    r'\bDROP\b|\bDELETE\b|\bTRUNCATE\b',  # SQL injection attempts (basic)
]

# Compile regex patterns for performance
COMPILED_DANGEROUS_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in DANGEROUS_PATTERNS]


class QueryValidator:
    """Validator for user queries with security and sanity checks."""
    
    def __init__(
        self,
        max_length: int = 2000,
        min_length: int = 1,
        allow_special_chars: bool = True,
        enable_xss_protection: bool = True,
    ):
        """Initialize query validator.
        
        Args:
            max_length: Maximum allowed query length
            min_length: Minimum allowed query length
            allow_special_chars: Whether to allow special characters
            enable_xss_protection: Enable XSS pattern detection
        """
        self.max_length = max_length
        self.min_length = min_length
        self.allow_special_chars = allow_special_chars
        self.enable_xss_protection = enable_xss_protection
    
    def validate(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate query and return (is_valid, error_message).
        
        Args:
            query: User query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not isinstance(query, str):
            return False, "Query must be a non-empty string"
        
        # Strip whitespace for length check
        stripped_query = query.strip()
        
        # Check length
        if len(stripped_query) < self.min_length:
            return False, f"Query too short (minimum: {self.min_length} characters)"
        
        if len(stripped_query) > self.max_length:
            return False, f"Query too long (maximum: {self.max_length} characters)"
        
        # Check for dangerous patterns
        if self.enable_xss_protection:
            for pattern in COMPILED_DANGEROUS_PATTERNS:
                if pattern.search(query):
                    return False, "Query contains potentially dangerous content"
        
        # Check for excessive special characters (potential abuse)
        if not self.allow_special_chars:
            if re.search(r'[^\w\s\.\,\?\!\-\']', query):
                return False, "Query contains disallowed special characters"
        
        # Check for excessive repetition (spam detection)
        if self._has_excessive_repetition(query):
            return False, "Query contains excessive repetition"
        
        return True, None
    
    def validate_or_raise(self, query: str) -> None:
        """Validate query and raise exception if invalid.
        
        Args:
            query: User query to validate
            
        Raises:
            InvalidQueryException: If query is invalid
            QueryTooLongException: If query exceeds maximum length
        """
        is_valid, error_message = self.validate(query)
        
        if not is_valid:
            if "too long" in error_message.lower():
                raise QueryTooLongException(len(query), self.max_length)
            else:
                raise InvalidQueryException(query, error_message)
    
    def sanitize(self, query: str) -> str:
        """Sanitize query by removing dangerous content.
        
        Args:
            query: User query to sanitize
            
        Returns:
            Sanitized query string
        """
        # Remove null bytes
        query = query.replace('\x00', '')
        
        # Remove control characters except newlines and tabs
        query = ''.join(
            char for char in query
            if ord(char) >= 32 or char in '\n\t'
        )
        
        # Remove dangerous patterns
        if self.enable_xss_protection:
            for pattern in COMPILED_DANGEROUS_PATTERNS:
                query = pattern.sub('', query)
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        # Trim to max length
        if len(query) > self.max_length:
            query = query[:self.max_length].rsplit(' ', 1)[0] + '...'
        
        return query.strip()
    
    @staticmethod
    def _has_excessive_repetition(text: str, threshold: float = 0.7) -> bool:
        """Check if text has excessive character repetition.
        
        Args:
            text: Text to check
            threshold: Ratio threshold for repetition
            
        Returns:
            True if excessive repetition detected
        """
        if len(text) < 10:
            return False
        
        # Count character frequencies
        char_counts: Dict[str, int] = {}
        for char in text.lower():
            if char.isalnum():
                char_counts[char] = char_counts.get(char, 0) + 1
        
        if not char_counts:
            return False
        
        # Check if any character appears more than threshold
        total_chars = sum(char_counts.values())
        max_freq = max(char_counts.values())
        
        return (max_freq / total_chars) > threshold


class ConfigValidator:
    """Validator for configuration values."""
    
    @staticmethod
    def validate_positive_int(value: Any, name: str) -> int:
        """Validate positive integer.
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            
        Returns:
            Validated integer
            
        Raises:
            ValidationException: If validation fails
        """
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise ValidationException(
                f"{name} must be an integer",
                details={"value": str(value), "name": name}
            )
        
        if int_value <= 0:
            raise ValidationException(
                f"{name} must be positive",
                details={"value": int_value, "name": name}
            )
        
        return int_value
    
    @staticmethod
    def validate_float_range(
        value: Any,
        name: str,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> float:
        """Validate float within range.
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated float
            
        Raises:
            ValidationException: If validation fails
        """
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValidationException(
                f"{name} must be a number",
                details={"value": str(value), "name": name}
            )
        
        if not (min_val <= float_value <= max_val):
            raise ValidationException(
                f"{name} must be between {min_val} and {max_val}",
                details={"value": float_value, "name": name, "range": [min_val, max_val]}
            )
        
        return float_value
    
    @staticmethod
    def validate_choice(value: Any, name: str, choices: List[Any]) -> Any:
        """Validate value is in allowed choices.
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            choices: List of allowed values
            
        Returns:
            Validated value
            
        Raises:
            ValidationException: If validation fails
        """
        if value not in choices:
            raise ValidationException(
                f"{name} must be one of: {', '.join(str(c) for c in choices)}",
                details={"value": str(value), "name": name, "choices": choices}
            )
        
        return value
    
    @staticmethod
    def validate_path(value: str, name: str, must_exist: bool = False) -> str:
        """Validate file path.
        
        Args:
            value: Path to validate
            name: Parameter name for error messages
            must_exist: Whether path must exist
            
        Returns:
            Validated path
            
        Raises:
            ValidationException: If validation fails
        """
        import os
        
        if not isinstance(value, str):
            raise ValidationException(
                f"{name} must be a string path",
                details={"value": str(value), "name": name}
            )
        
        if must_exist and not os.path.exists(value):
            raise ValidationException(
                f"{name} path does not exist",
                details={"value": value, "name": name}
            )
        
        # Check for path traversal attempts
        if '..' in value or value.startswith('/etc') or value.startswith('/sys'):
            raise ValidationException(
                f"{name} contains invalid path components",
                details={"value": value, "name": name}
            )
        
        return value


class EvidenceValidator:
    """Validator for retrieved evidence and documents."""
    
    @staticmethod
    def validate_evidence_item(item: Dict[str, Any]) -> bool:
        """Validate evidence item structure.
        
        Args:
            item: Evidence item to validate
            
        Returns:
            True if valid
        """
        required_fields = ['doc_id', 'text']
        
        if not isinstance(item, dict):
            return False
        
        for field in required_fields:
            if field not in item:
                return False
        
        if not isinstance(item['text'], str) or not item['text'].strip():
            return False
        
        if 'score' in item:
            try:
                score = float(item['score'])
                if not (0.0 <= score <= 1.0):
                    return False
            except (TypeError, ValueError):
                return False
        
        return True
    
    @staticmethod
    def validate_evidence_list(
        evidence: List[Dict[str, Any]],
        min_items: int = 0,
        max_items: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate list of evidence items.
        
        Args:
            evidence: List of evidence items
            min_items: Minimum required items
            max_items: Maximum allowed items
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(evidence, list):
            return False, "Evidence must be a list"
        
        if len(evidence) < min_items:
            return False, f"Evidence list must contain at least {min_items} items"
        
        if max_items and len(evidence) > max_items:
            return False, f"Evidence list cannot exceed {max_items} items"
        
        for i, item in enumerate(evidence):
            if not EvidenceValidator.validate_evidence_item(item):
                return False, f"Invalid evidence item at index {i}"
        
        return True, None


def sanitize_dict_for_logging(data: Dict[str, Any], sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Sanitize dictionary for logging by redacting sensitive fields.
    
    Args:
        data: Dictionary to sanitize
        sensitive_keys: List of keys to redact (default: common sensitive keys)
        
    Returns:
        Sanitized dictionary
    """
    if sensitive_keys is None:
        sensitive_keys = [
            'password', 'token', 'api_key', 'secret', 'credential',
            'auth', 'session_id', 'user_id', 'email'
        ]
    
    sanitized = {}
    
    for key, value in data.items():
        # Check if key is sensitive
        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
            sanitized[key] = '***REDACTED***'
        elif isinstance(value, dict):
            # Recursively sanitize nested dicts
            sanitized[key] = sanitize_dict_for_logging(value, sensitive_keys)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Sanitize list of dicts
            sanitized[key] = [sanitize_dict_for_logging(item, sensitive_keys) for item in value]
        else:
            sanitized[key] = value
    
    return sanitized
