"""
Comprehensive error handling and validation system for schema matching.
"""
import logging
import traceback
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError

class ErrorSeverity(Enum):
    """Error severity levels for categorizing issues."""
    CRITICAL = "CRITICAL"  # System cannot continue
    ERROR = "ERROR"        # Feature fails but system continues  
    WARNING = "WARNING"    # Suboptimal behavior but functional
    INFO = "INFO"         # Informational messages

class SchemaMatchingError(Exception):
    """Base exception for all schema matching errors."""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.severity = ErrorSeverity.ERROR

class InvalidSchemaError(SchemaMatchingError):
    """Raised when schema is invalid, empty, or malformed."""
    def __init__(self, message: str, schema_path: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "INVALID_SCHEMA", details)
        self.schema_path = schema_path
        self.severity = ErrorSeverity.CRITICAL

class DataValidationError(SchemaMatchingError):
    """Raised when data validation fails."""
    def __init__(self, message: str, column: str = None, row_index: int = None, details: Dict[str, Any] = None):
        super().__init__(message, "DATA_VALIDATION_ERROR", details)
        self.column = column
        self.row_index = row_index
        self.severity = ErrorSeverity.ERROR

class EmbeddingError(SchemaMatchingError):
    """Raised when embedding generation or processing fails."""
    def __init__(self, message: str, text: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "EMBEDDING_ERROR", details)
        self.text = text
        self.severity = ErrorSeverity.ERROR

class TransformationError(SchemaMatchingError):
    """Raised when data transformation fails."""
    def __init__(self, message: str, source_column: str = None, target_column: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "TRANSFORMATION_ERROR", details)
        self.source_column = source_column
        self.target_column = target_column
        self.severity = ErrorSeverity.WARNING

class APIError(SchemaMatchingError):
    """Raised when external API calls fail."""
    def __init__(self, message: str, api_name: str = None, status_code: int = None, details: Dict[str, Any] = None):
        super().__init__(message, "API_ERROR", details)
        self.api_name = api_name
        self.status_code = status_code
        self.severity = ErrorSeverity.ERROR

class ConfigurationError(SchemaMatchingError):
    """Raised when configuration is invalid or missing."""
    def __init__(self, message: str, config_key: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key
        self.severity = ErrorSeverity.CRITICAL

class ErrorTracker:
    """Tracks and aggregates errors across the pipeline."""
    
    def __init__(self):
        self.errors: List[SchemaMatchingError] = []
        self.warnings: List[SchemaMatchingError] = []
        self.info_messages: List[str] = []
    
    def add_error(self, error: SchemaMatchingError):
        """Add an error to the tracker."""
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.ERROR]:
            self.errors.append(error)
            logging.error(f"[{error.error_code}] {error.message}")
        elif error.severity == ErrorSeverity.WARNING:
            self.warnings.append(error)
            logging.warning(f"[{error.error_code}] {error.message}")
        else:
            self.info_messages.append(error.message)
            logging.info(f"[{error.error_code}] {error.message}")
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors occurred."""
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.errors)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors and warnings."""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "critical_errors": [e for e in self.errors if e.severity == ErrorSeverity.CRITICAL],
            "error_codes": [e.error_code for e in self.errors],
            "warning_codes": [w.error_code for w in self.warnings],
        }
    
    def clear(self):
        """Clear all tracked errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
        self.info_messages.clear()

# Global error tracker instance
error_tracker = ErrorTracker()

def handle_errors(error_tracker: ErrorTracker = None, reraise: bool = False):
    """
    Decorator for comprehensive error handling with context tracking.
    
    Args:
        error_tracker: ErrorTracker instance to use (uses global if None)
        reraise: Whether to reraise exceptions after logging
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = error_tracker or globals()['error_tracker']
            
            try:
                return func(*args, **kwargs)
            except SchemaMatchingError as e:
                tracker.add_error(e)
                if reraise or e.severity == ErrorSeverity.CRITICAL:
                    raise
                return None
            except ValidationError as e:
                error = DataValidationError(
                    f"Pydantic validation failed in {func.__name__}: {str(e)}",
                    details={"validation_errors": e.errors()}
                )
                tracker.add_error(error)
                if reraise:
                    raise error
                return None
            except pd.errors.ParserError as e:
                error = DataValidationError(
                    f"Pandas parsing error in {func.__name__}: {str(e)}",
                    details={"original_error": str(e)}
                )
                tracker.add_error(error)
                if reraise:
                    raise error
                return None
            except FileNotFoundError as e:
                error = ConfigurationError(
                    f"Required file not found in {func.__name__}: {str(e)}",
                    details={"file_path": str(e)}
                )
                tracker.add_error(error)
                if reraise:
                    raise error
                return None
            except Exception as e:
                # Catch-all for unexpected errors
                error = SchemaMatchingError(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    "UNEXPECTED_ERROR",
                    {
                        "function": func.__name__,
                        "args": str(args)[:200] if args else None,
                        "kwargs": str(kwargs)[:200] if kwargs else None,
                        "traceback": traceback.format_exc()
                    }
                )
                error.severity = ErrorSeverity.CRITICAL
                tracker.add_error(error)
                
                if reraise:
                    raise error
                return None
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracker = error_tracker or globals()['error_tracker']
            
            try:
                return await func(*args, **kwargs)
            except SchemaMatchingError as e:
                tracker.add_error(e)
                if reraise or e.severity == ErrorSeverity.CRITICAL:
                    raise
                return None
            except Exception as e:
                error = SchemaMatchingError(
                    f"Unexpected async error in {func.__name__}: {str(e)}",
                    "ASYNC_ERROR",
                    {"function": func.__name__, "traceback": traceback.format_exc()}
                )
                error.severity = ErrorSeverity.CRITICAL
                tracker.add_error(error)
                
                if reraise:
                    raise error
                return None
        
        # Return async wrapper for coroutines, sync wrapper otherwise
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

class SchemaValidator:
    """Validates schema objects and data structures."""
    
    @staticmethod
    def validate_object_schema(schema, schema_path: str = None) -> bool:
        """Validate an ObjectSchema instance."""
        if schema is None:
            raise InvalidSchemaError("Schema is None", schema_path)
        
        if not hasattr(schema, 'properties') or not schema.properties:
            raise InvalidSchemaError("Schema has no properties", schema_path)
        
        if len(schema.properties) == 0:
            raise InvalidSchemaError("Schema properties are empty", schema_path)
        
        # Validate each property
        for prop_name, prop_schema in schema.properties.items():
            if not hasattr(prop_schema, 'type'):
                raise InvalidSchemaError(
                    f"Property '{prop_name}' missing type", 
                    schema_path,
                    {"property": prop_name}
                )
            
            valid_types = {"string", "integer", "number", "boolean", "object", "array"}
            if prop_schema.type not in valid_types:
                raise InvalidSchemaError(
                    f"Property '{prop_name}' has invalid type: {prop_schema.type}",
                    schema_path,
                    {"property": prop_name, "invalid_type": prop_schema.type}
                )
        
        return True
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, required_columns: List[str] = None) -> bool:
        """Validate a pandas DataFrame."""
        if df is None:
            raise DataValidationError("DataFrame is None")
        
        if df.empty and min_rows > 0:
            raise DataValidationError(f"DataFrame is empty, expected at least {min_rows} rows")
        
        if len(df) < min_rows:
            raise DataValidationError(
                f"DataFrame has {len(df)} rows, expected at least {min_rows}",
                details={"actual_rows": len(df), "min_required": min_rows}
            )
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise DataValidationError(
                    f"DataFrame missing required columns: {missing_columns}",
                    details={"missing_columns": list(missing_columns), "available_columns": list(df.columns)}
                )
        
        # Check for completely null columns
        null_columns = [col for col in df.columns if df[col].isna().all()]
        if null_columns:
            error_tracker.add_error(TransformationError(
                f"Found columns with all null values: {null_columns}",
                details={"null_columns": null_columns}
            ))
        
        return True
    
    @staticmethod
    def validate_column_mapping(mapping: Dict[str, Any], source_columns: List[str], target_columns: List[str]) -> bool:
        """Validate a column mapping dictionary."""
        if not mapping:
            raise DataValidationError("Column mapping is empty")
        
        # Check target columns exist
        invalid_targets = set(mapping.keys()) - set(target_columns)
        if invalid_targets:
            raise DataValidationError(
                f"Mapping contains invalid target columns: {invalid_targets}",
                details={"invalid_targets": list(invalid_targets), "valid_targets": target_columns}
            )
        
        # Check source columns exist (allowing for null mappings)
        invalid_sources = []
        for target, source_info in mapping.items():
            if isinstance(source_info, tuple):
                source_col = source_info[0]
            else:
                source_col = source_info
            
            if source_col is not None and source_col not in source_columns and source_col != "null":
                invalid_sources.append((target, source_col))
        
        if invalid_sources:
            raise DataValidationError(
                f"Mapping contains invalid source columns: {invalid_sources}",
                details={"invalid_mappings": invalid_sources, "valid_sources": source_columns}
            )
        
        return True

class DataSanitizer:
    """Sanitizes and cleans data before processing."""
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and sanitize a DataFrame."""
        if df.empty:
            return df
        
        # Remove completely empty rows
        original_rows = len(df)
        df = df.dropna(how='all')
        removed_rows = original_rows - len(df)
        
        if removed_rows > 0:
            error_tracker.add_error(TransformationError(
                f"Removed {removed_rows} completely empty rows",
                details={"removed_rows": removed_rows, "original_rows": original_rows}
            ))
        
        # Clean column names
        original_columns = list(df.columns)
        df.columns = df.columns.str.strip()  # Remove whitespace
        df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)  # Replace special chars
        
        renamed_columns = {old: new for old, new in zip(original_columns, df.columns) if old != new}
        if renamed_columns:
            error_tracker.add_error(TransformationError(
                f"Sanitized column names: {renamed_columns}",
                details={"renamed_columns": renamed_columns}
            ))
        
        return df
    
    @staticmethod
    def sanitize_text_for_embedding(text: str) -> str:
        """Clean text before sending to embedding API."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit length to prevent API errors
        max_length = 8000  # Conservative limit for most embedding APIs
        if len(text) > max_length:
            text = text[:max_length]
            error_tracker.add_error(TransformationError(
                f"Truncated text to {max_length} characters for embedding",
                details={"original_length": len(text), "truncated_length": max_length}
            ))
        
        return text

def validate_configuration(config: Dict[str, Any], required_keys: List[str] = None) -> bool:
    """Validate configuration dictionary."""
    if not config:
        raise ConfigurationError("Configuration is empty or None")
    
    if required_keys:
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            raise ConfigurationError(
                f"Configuration missing required keys: {missing_keys}",
                details={"missing_keys": list(missing_keys), "available_keys": list(config.keys())}
            )
    
    return True

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    try:
        if denominator == 0:
            error_tracker.add_error(TransformationError(
                "Division by zero avoided",
                details={"numerator": numerator, "default_returned": default}
            ))
            return default
        return numerator / denominator
    except (TypeError, ValueError) as e:
        error_tracker.add_error(TransformationError(
            f"Invalid division operation: {e}",
            details={"numerator": numerator, "denominator": denominator, "default_returned": default}
        ))
        return default

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if pd.isna(value) or value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        error_tracker.add_error(TransformationError(
            f"Failed to convert '{value}' to float, using default {default}",
            details={"original_value": str(value), "default_used": default}
        ))
        return default

def safe_json_load(file_path: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """Safely load JSON file with error handling."""
    import json
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ConfigurationError(f"JSON file not found: {file_path}", config_key=file_path)
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Invalid JSON in file: {file_path}",
            config_key=file_path,
            details={"json_error": str(e)}
        )
    except Exception as e:
        raise ConfigurationError(
            f"Unexpected error loading JSON file: {file_path}",
            config_key=file_path,
            details={"error": str(e)}
        )

# Context manager for error tracking
class ErrorContext:
    """Context manager for tracking errors within a specific operation."""
    
    def __init__(self, operation_name: str, fail_fast: bool = False):
        self.operation_name = operation_name
        self.fail_fast = fail_fast
        self.local_tracker = ErrorTracker()
    
    def __enter__(self):
        return self.local_tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if issubclass(exc_type, SchemaMatchingError):
                self.local_tracker.add_error(exc_val)
            else:
                error = SchemaMatchingError(
                    f"Unexpected error in {self.operation_name}: {str(exc_val)}",
                    "CONTEXT_ERROR",
                    {"operation": self.operation_name, "exception_type": exc_type.__name__}
                )
                error.severity = ErrorSeverity.CRITICAL
                self.local_tracker.add_error(error)
        
        # Merge local errors into global tracker
        for error in self.local_tracker.errors + self.local_tracker.warnings:
            error_tracker.add_error(error)
        
        # Suppress exceptions unless fail_fast is True
        if self.fail_fast and exc_type is not None:
            return False  # Re-raise exception
        return True  # Suppress exception
