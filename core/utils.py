"""Common utility functions used across the RAG Agent project.

This module contains shared utilities to avoid code duplication:
- File I/O helpers
- JSON handling
- Environment variable parsing
- Logging configuration
"""
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union


def get_env_bool(key: str, default: bool = False) -> bool:
    """Parse environment variable as boolean.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Boolean value from environment or default
    """
    value = os.getenv(key, str(int(default)))
    return value.lower() in ("1", "true", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Parse environment variable as integer.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Integer value from environment or default
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning(f"Invalid integer for {key}='{value}', using default {default}")
        return default


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: Union[str, Path], default: Any = None) -> Any:
    """Load JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Parsed JSON data or default value
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logging.debug(f"JSON file not found: {filepath}, using default")
        return default
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {filepath}: {e}")
        return default
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {e}")
        return default


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> bool:
    """Save data to JSON file with error handling.
    
    Args:
        data: Data to serialize
        filepath: Path to save JSON file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {e}")
        return False


def setup_logger(
    name: str,
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Configure and return a logger instance.
    
    Args:
        name: Logger name
        level: Logging level (int or string)
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
