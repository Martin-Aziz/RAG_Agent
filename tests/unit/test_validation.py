"""Unit tests for validation module."""
import pytest
from core.validation import (
    QueryValidator,
    ConfigValidator,
    EvidenceValidator,
    sanitize_dict_for_logging
)
from core.exceptions import InvalidQueryException, QueryTooLongException, ValidationException


class TestQueryValidator:
    """Test cases for QueryValidator."""
    
    def test_valid_query(self):
        """Test validation of valid query."""
        validator = QueryValidator()
        is_valid, error = validator.validate("What is Python?")
        assert is_valid
        assert error is None
    
    def test_empty_query(self):
        """Test validation of empty query."""
        validator = QueryValidator()
        is_valid, error = validator.validate("")
        assert not is_valid
        assert "non-empty" in error
    
    def test_query_too_long(self):
        """Test validation of query exceeding max length."""
        validator = QueryValidator(max_length=100)
        long_query = "x" * 101
        is_valid, error = validator.validate(long_query)
        assert not is_valid
        assert "too long" in error.lower()
    
    def test_query_too_short(self):
        """Test validation of query below min length."""
        validator = QueryValidator(min_length=5)
        is_valid, error = validator.validate("hi")
        assert not is_valid
        assert "too short" in error.lower()
    
    def test_xss_detection(self):
        """Test XSS pattern detection."""
        validator = QueryValidator(enable_xss_protection=True)
        xss_query = "<script>alert('xss')</script>"
        is_valid, error = validator.validate(xss_query)
        assert not is_valid
        assert "dangerous" in error.lower()
    
    def test_javascript_protocol_detection(self):
        """Test JavaScript protocol detection."""
        validator = QueryValidator(enable_xss_protection=True)
        js_query = "javascript:void(0)"
        is_valid, error = validator.validate(js_query)
        assert not is_valid
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        validator = QueryValidator(enable_xss_protection=True)
        sql_query = "SELECT * FROM users; DROP TABLE users;"
        is_valid, error = validator.validate(sql_query)
        assert not is_valid
    
    def test_excessive_repetition(self):
        """Test excessive repetition detection."""
        validator = QueryValidator()
        spam_query = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        is_valid, error = validator.validate(spam_query)
        assert not is_valid
        assert "repetition" in error.lower()
    
    def test_sanitize_removes_dangerous_content(self):
        """Test query sanitization."""
        validator = QueryValidator(enable_xss_protection=True)
        dirty_query = "<script>alert('test')</script> What is Python?"
        clean_query = validator.sanitize(dirty_query)
        assert "<script>" not in clean_query
        assert "What is Python?" in clean_query
    
    def test_sanitize_normalizes_whitespace(self):
        """Test whitespace normalization."""
        validator = QueryValidator()
        query = "What   is    Python?"
        clean_query = validator.sanitize(query)
        assert clean_query == "What is Python?"
    
    def test_sanitize_removes_null_bytes(self):
        """Test null byte removal."""
        validator = QueryValidator()
        query = "What\x00is Python?"
        clean_query = validator.sanitize(query)
        assert "\x00" not in clean_query
    
    def test_validate_or_raise_valid(self):
        """Test validate_or_raise with valid query."""
        validator = QueryValidator()
        # Should not raise
        validator.validate_or_raise("What is Python?")
    
    def test_validate_or_raise_invalid(self):
        """Test validate_or_raise with invalid query."""
        validator = QueryValidator()
        with pytest.raises(InvalidQueryException):
            validator.validate_or_raise("")
    
    def test_validate_or_raise_too_long(self):
        """Test validate_or_raise with too long query."""
        validator = QueryValidator(max_length=10)
        with pytest.raises(QueryTooLongException):
            validator.validate_or_raise("This is a very long query")


class TestConfigValidator:
    """Test cases for ConfigValidator."""
    
    def test_validate_positive_int_valid(self):
        """Test positive integer validation with valid value."""
        result = ConfigValidator.validate_positive_int(10, "test_param")
        assert result == 10
    
    def test_validate_positive_int_zero(self):
        """Test positive integer validation with zero."""
        with pytest.raises(ValidationException):
            ConfigValidator.validate_positive_int(0, "test_param")
    
    def test_validate_positive_int_negative(self):
        """Test positive integer validation with negative value."""
        with pytest.raises(ValidationException):
            ConfigValidator.validate_positive_int(-5, "test_param")
    
    def test_validate_positive_int_string(self):
        """Test positive integer validation with string."""
        result = ConfigValidator.validate_positive_int("10", "test_param")
        assert result == 10
    
    def test_validate_positive_int_invalid_string(self):
        """Test positive integer validation with invalid string."""
        with pytest.raises(ValidationException):
            ConfigValidator.validate_positive_int("abc", "test_param")
    
    def test_validate_float_range_valid(self):
        """Test float range validation with valid value."""
        result = ConfigValidator.validate_float_range(0.5, "test_param", 0.0, 1.0)
        assert result == 0.5
    
    def test_validate_float_range_below_min(self):
        """Test float range validation below minimum."""
        with pytest.raises(ValidationException):
            ConfigValidator.validate_float_range(-0.1, "test_param", 0.0, 1.0)
    
    def test_validate_float_range_above_max(self):
        """Test float range validation above maximum."""
        with pytest.raises(ValidationException):
            ConfigValidator.validate_float_range(1.5, "test_param", 0.0, 1.0)
    
    def test_validate_choice_valid(self):
        """Test choice validation with valid value."""
        result = ConfigValidator.validate_choice("a", "test_param", ["a", "b", "c"])
        assert result == "a"
    
    def test_validate_choice_invalid(self):
        """Test choice validation with invalid value."""
        with pytest.raises(ValidationException):
            ConfigValidator.validate_choice("d", "test_param", ["a", "b", "c"])
    
    def test_validate_path_traversal(self):
        """Test path validation rejects traversal."""
        with pytest.raises(ValidationException):
            ConfigValidator.validate_path("../../../etc/passwd", "test_path")
    
    def test_validate_path_system_dirs(self):
        """Test path validation rejects system directories."""
        with pytest.raises(ValidationException):
            ConfigValidator.validate_path("/etc/passwd", "test_path")


class TestEvidenceValidator:
    """Test cases for EvidenceValidator."""
    
    def test_validate_evidence_item_valid(self):
        """Test validation of valid evidence item."""
        item = {
            "doc_id": "doc1",
            "text": "This is some evidence",
            "score": 0.8
        }
        assert EvidenceValidator.validate_evidence_item(item)
    
    def test_validate_evidence_item_missing_doc_id(self):
        """Test validation fails for missing doc_id."""
        item = {"text": "This is some evidence"}
        assert not EvidenceValidator.validate_evidence_item(item)
    
    def test_validate_evidence_item_empty_text(self):
        """Test validation fails for empty text."""
        item = {"doc_id": "doc1", "text": ""}
        assert not EvidenceValidator.validate_evidence_item(item)
    
    def test_validate_evidence_item_invalid_score(self):
        """Test validation fails for invalid score."""
        item = {"doc_id": "doc1", "text": "Evidence", "score": 1.5}
        assert not EvidenceValidator.validate_evidence_item(item)
    
    def test_validate_evidence_list_valid(self):
        """Test validation of valid evidence list."""
        evidence = [
            {"doc_id": "doc1", "text": "Evidence 1"},
            {"doc_id": "doc2", "text": "Evidence 2"}
        ]
        is_valid, error = EvidenceValidator.validate_evidence_list(evidence)
        assert is_valid
        assert error is None
    
    def test_validate_evidence_list_too_short(self):
        """Test validation fails for list below minimum."""
        evidence = []
        is_valid, error = EvidenceValidator.validate_evidence_list(
            evidence,
            min_items=1
        )
        assert not is_valid
        assert "at least" in error
    
    def test_validate_evidence_list_too_long(self):
        """Test validation fails for list above maximum."""
        evidence = [
            {"doc_id": f"doc{i}", "text": f"Evidence {i}"}
            for i in range(10)
        ]
        is_valid, error = EvidenceValidator.validate_evidence_list(
            evidence,
            max_items=5
        )
        assert not is_valid
        assert "exceed" in error
    
    def test_validate_evidence_list_invalid_item(self):
        """Test validation fails for invalid item in list."""
        evidence = [
            {"doc_id": "doc1", "text": "Evidence 1"},
            {"doc_id": "doc2"},  # Missing text
        ]
        is_valid, error = EvidenceValidator.validate_evidence_list(evidence)
        assert not is_valid
        assert "Invalid evidence item" in error


class TestSanitizeDict:
    """Test cases for sanitize_dict_for_logging."""
    
    def test_sanitize_sensitive_keys(self):
        """Test sanitization of sensitive keys."""
        data = {
            "query": "What is Python?",
            "password": "secret123",
            "api_key": "abc123"
        }
        sanitized = sanitize_dict_for_logging(data)
        assert sanitized["query"] == "What is Python?"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
    
    def test_sanitize_nested_dict(self):
        """Test sanitization of nested dictionaries."""
        data = {
            "user": {
                "name": "John",
                "password": "secret"
            }
        }
        sanitized = sanitize_dict_for_logging(data)
        assert sanitized["user"]["name"] == "John"
        assert sanitized["user"]["password"] == "***REDACTED***"
    
    def test_sanitize_list_of_dicts(self):
        """Test sanitization of list of dictionaries."""
        data = {
            "users": [
                {"name": "John", "email": "john@example.com"},
                {"name": "Jane", "email": "jane@example.com"}
            ]
        }
        sanitized = sanitize_dict_for_logging(data)
        assert sanitized["users"][0]["name"] == "John"
        assert sanitized["users"][0]["email"] == "***REDACTED***"
    
    def test_sanitize_custom_keys(self):
        """Test sanitization with custom sensitive keys."""
        data = {
            "query": "test",
            "custom_secret": "sensitive"
        }
        sanitized = sanitize_dict_for_logging(data, sensitive_keys=["custom_secret"])
        assert sanitized["query"] == "test"
        assert sanitized["custom_secret"] == "***REDACTED***"
