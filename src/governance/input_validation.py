"""
Input validation and sanitization for RAG chatbot.
"""
import re
from typing import Dict, List, Any, Optional, Union
import logging
import bleach
from pathlib import Path

logger = logging.getLogger(__name__)


class InputValidator:
    """Validate and sanitize user inputs for security and safety."""
    
    def __init__(self):
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            # SQL injection patterns
            r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bDROP\b|\bCREATE\b)",
            # Command injection patterns
            r"(\$\(|\`|;|\||&|\n|\r)",
            # Path traversal patterns
            r"(\.\.\/|\.\.\\)",
            # Script injection patterns
            r"(<script>|</script>|javascript:|vbscript:)",
            # Code execution patterns
            r"(\bexec\b|\beval\b|\b__import__\b|\bgetattr\b)",
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
        
        # Maximum allowed lengths
        self.max_lengths = {
            'query': 2000,
            'filename': 255,
            'filepath': 1000,
            'general_text': 10000
        }
        
        # Allowed file extensions
        self.allowed_extensions = {'.txt', '.pdf', '.docx', '.md', '.csv'}
        
        # HTML tags to allow (very restrictive)
        self.allowed_tags = []
        self.allowed_attributes = {}
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate a user query for safety and format."""
        try:
            result = {
                'is_valid': True,
                'sanitized_query': query,
                'warnings': [],
                'errors': []
            }
            
            # Check if query is empty or None
            if not query or not query.strip():
                result['is_valid'] = False
                result['errors'].append("Query cannot be empty")
                return result
            
            # Check length
            if len(query) > self.max_lengths['query']:
                result['is_valid'] = False
                result['errors'].append(f"Query exceeds maximum length of {self.max_lengths['query']} characters")
                return result
            
            # Check for dangerous patterns
            dangerous_found = []
            for pattern in self.compiled_patterns:
                if pattern.search(query):
                    dangerous_found.append(pattern.pattern)
            
            if dangerous_found:
                result['is_valid'] = False
                result['errors'].append(f"Potentially dangerous patterns detected: {dangerous_found}")
                return result
            
            # Sanitize HTML content
            sanitized = bleach.clean(query, tags=self.allowed_tags, attributes=self.allowed_attributes)
            
            if sanitized != query:
                result['warnings'].append("HTML content was sanitized")
                result['sanitized_query'] = sanitized
            
            # Additional content checks
            self._check_content_quality(sanitized, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return {
                'is_valid': False,
                'sanitized_query': '',
                'warnings': [],
                'errors': [f"Validation error: {str(e)}"]
            }
    
    def validate_file_path(self, file_path: str) -> Dict[str, Any]:
        """Validate file path for security and accessibility."""
        try:
            result = {
                'is_valid': True,
                'normalized_path': file_path,
                'warnings': [],
                'errors': []
            }
            
            # Check if path is empty
            if not file_path or not file_path.strip():
                result['is_valid'] = False
                result['errors'].append("File path cannot be empty")
                return result
            
            # Normalize path
            try:
                path = Path(file_path).resolve()
                result['normalized_path'] = str(path)
            except Exception as e:
                result['is_valid'] = False
                result['errors'].append(f"Invalid path format: {str(e)}")
                return result
            
            # Check for path traversal attempts
            if '..' in file_path:
                result['warnings'].append("Path traversal patterns detected and normalized")
            
            # Check file extension
            if path.suffix.lower() not in self.allowed_extensions:
                result['is_valid'] = False
                result['errors'].append(f"File extension '{path.suffix}' not allowed. Allowed: {self.allowed_extensions}")
                return result
            
            # Check if file exists (if it should)
            if not path.exists():
                result['warnings'].append("File does not exist")
            
            # Check file size if exists
            if path.exists() and path.is_file():
                size = path.stat().st_size
                max_size = 50 * 1024 * 1024  # 50MB
                if size > max_size:
                    result['is_valid'] = False
                    result['errors'].append(f"File size ({size}) exceeds maximum allowed ({max_size} bytes)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating file path: {e}")
            return {
                'is_valid': False,
                'normalized_path': '',
                'warnings': [],
                'errors': [f"Path validation error: {str(e)}"]
            }
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters."""
        try:
            result = {
                'is_valid': True,
                'sanitized_config': config.copy(),
                'warnings': [],
                'errors': []
            }
            
            # Define valid configuration keys and their constraints
            valid_configs = {
                'chunk_size': {'type': int, 'min': 100, 'max': 5000, 'default': 1000},
                'chunk_overlap': {'type': int, 'min': 0, 'max': 500, 'default': 200},
                'top_k': {'type': int, 'min': 1, 'max': 20, 'default': 5},
                'temperature': {'type': float, 'min': 0.0, 'max': 2.0, 'default': 0.7},
                'max_length': {'type': int, 'min': 100, 'max': 8192, 'default': 2048},
                'model_name': {'type': str, 'allowed': ['mistral-7b', 'llama2-7b', 'falcon-7b'], 'default': 'mistral-7b'},
                'vector_store': {'type': str, 'allowed': ['faiss', 'chroma'], 'default': 'chroma'},
                'embedding_model': {'type': str, 'allowed': ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'], 'default': 'all-MiniLM-L6-v2'}
            }
            
            # Validate each configuration parameter
            for key, value in config.items():
                if key not in valid_configs:
                    result['warnings'].append(f"Unknown configuration key: {key}")
                    continue
                
                constraints = valid_configs[key]
                
                # Type validation
                expected_type = constraints['type']
                if not isinstance(value, expected_type):
                    try:
                        # Try to convert
                        if expected_type == int:
                            value = int(value)
                        elif expected_type == float:
                            value = float(value)
                        elif expected_type == str:
                            value = str(value)
                        result['sanitized_config'][key] = value
                        result['warnings'].append(f"Converted {key} to {expected_type.__name__}")
                    except (ValueError, TypeError):
                        result['is_valid'] = False
                        result['errors'].append(f"Invalid type for {key}: expected {expected_type.__name__}")
                        continue
                
                # Range validation for numeric types
                if expected_type in [int, float]:
                    if 'min' in constraints and value < constraints['min']:
                        result['is_valid'] = False
                        result['errors'].append(f"{key} value {value} below minimum {constraints['min']}")
                    if 'max' in constraints and value > constraints['max']:
                        result['is_valid'] = False
                        result['errors'].append(f"{key} value {value} above maximum {constraints['max']}")
                
                # Allowed values validation
                if 'allowed' in constraints:
                    if value not in constraints['allowed']:
                        result['is_valid'] = False
                        result['errors'].append(f"Invalid value for {key}: {value}. Allowed: {constraints['allowed']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return {
                'is_valid': False,
                'sanitized_config': {},
                'warnings': [],
                'errors': [f"Configuration validation error: {str(e)}"]
            }
    
    def _check_content_quality(self, content: str, result: Dict[str, Any]) -> None:
        """Check content quality and add warnings if needed."""
        # Check for repetitive content
        words = content.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                result['warnings'].append("Content appears highly repetitive")
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', content)) / len(content)
        if special_char_ratio > 0.3:
            result['warnings'].append("High ratio of special characters detected")
        
        # Check for very long words (potential gibberish)
        long_words = [word for word in words if len(word) > 50]
        if long_words:
            result['warnings'].append(f"Unusually long words detected: {len(long_words)}")
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove or replace dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure filename is not too long
        if len(sanitized) > self.max_lengths['filename']:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            max_name_len = self.max_lengths['filename'] - len(ext)
            sanitized = name[:max_name_len] + ext
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = "unnamed_file"
        
        return sanitized
    
    def is_safe_content(self, content: str) -> bool:
        """Quick check if content appears safe."""
        try:
            validation_result = self.validate_query(content)
            return validation_result['is_valid']
        except Exception:
            return False
