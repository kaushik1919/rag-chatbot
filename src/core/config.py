"""
Configuration management module.

Handles application settings, environment variables, and configuration validation.
Provides centralized configuration management for the RAG chatbot system.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    use_quantization: bool = False
    quantization_bits: int = 8


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""
    type: str = "faiss"  # faiss or chroma
    dimension: int = 384
    similarity_threshold: float = 0.7
    max_results: int = 5
    persist_directory: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "all-MiniLM-L6-v2"
    max_seq_length: int = 256
    batch_size: int = 32


@dataclass
class SecurityConfig:
    """Configuration for security and governance."""
    enable_content_filter: bool = True
    enable_input_validation: bool = True
    enable_rate_limiting: bool = True
    max_input_length: int = 2048
    allowed_file_types: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".docx", ".md"])
    max_file_size_mb: int = 10


@dataclass
class WebConfig:
    """Configuration for web interfaces."""
    streamlit_port: int = 8501
    fastapi_port: int = 8000
    host: str = "0.0.0.0"
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    web: WebConfig = field(default_factory=WebConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Data directories
    data_dir: str = "./data"
    documents_dir: str = "./data/documents"
    vector_store_dir: str = "./data/vector_store"
    logs_dir: str = "./logs"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            config = cls()
            config._update_from_dict(config_data)
            return config
            
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_path} not found. Using defaults.")
            return cls()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return cls()
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        config = cls()
        
        # Model configuration
        if os.getenv('MODEL_NAME'):
            config.model.name = os.getenv('MODEL_NAME')
        if os.getenv('MODEL_DEVICE'):
            config.model.device = os.getenv('MODEL_DEVICE')
        if os.getenv('MODEL_MAX_LENGTH'):
            config.model.max_length = int(os.getenv('MODEL_MAX_LENGTH'))
        if os.getenv('MODEL_TEMPERATURE'):
            config.model.temperature = float(os.getenv('MODEL_TEMPERATURE'))
        if os.getenv('USE_QUANTIZATION'):
            config.model.use_quantization = os.getenv('USE_QUANTIZATION').lower() == 'true'
        
        # Vector store configuration
        if os.getenv('VECTOR_STORE_TYPE'):
            config.vector_store.type = os.getenv('VECTOR_STORE_TYPE')
        if os.getenv('VECTOR_STORE_DIMENSION'):
            config.vector_store.dimension = int(os.getenv('VECTOR_STORE_DIMENSION'))
        if os.getenv('VECTOR_STORE_PERSIST_DIR'):
            config.vector_store.persist_directory = os.getenv('VECTOR_STORE_PERSIST_DIR')
        
        # Web configuration
        if os.getenv('STREAMLIT_PORT'):
            config.web.streamlit_port = int(os.getenv('STREAMLIT_PORT'))
        if os.getenv('FASTAPI_PORT'):
            config.web.fastapi_port = int(os.getenv('FASTAPI_PORT'))
        if os.getenv('WEB_HOST'):
            config.web.host = os.getenv('WEB_HOST')
        if os.getenv('DEBUG'):
            config.web.debug = os.getenv('DEBUG').lower() == 'true'
        
        # Data directories
        if os.getenv('DATA_DIR'):
            config.data_dir = os.getenv('DATA_DIR')
        if os.getenv('DOCUMENTS_DIR'):
            config.documents_dir = os.getenv('DOCUMENTS_DIR')
        if os.getenv('VECTOR_STORE_DIR'):
            config.vector_store_dir = os.getenv('VECTOR_STORE_DIR')
        if os.getenv('LOGS_DIR'):
            config.logs_dir = os.getenv('LOGS_DIR')
        
        return config
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
            elif hasattr(self, section_name):
                setattr(self, section_name, section_data)
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.documents_dir,
            self.vector_store_dir,
            self.logs_dir
        ]
        
        if self.vector_store.persist_directory:
            directories.append(self.vector_store.persist_directory)
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate model configuration
            if not self.model.name:
                raise ValueError("Model name cannot be empty")
            
            if self.model.max_length <= 0:
                raise ValueError("Model max_length must be positive")
            
            if not 0 <= self.model.temperature <= 2:
                raise ValueError("Model temperature must be between 0 and 2")
            
            # Validate vector store configuration
            if self.vector_store.type not in ["faiss", "chroma"]:
                raise ValueError("Vector store type must be 'faiss' or 'chroma'")
            
            if self.vector_store.dimension <= 0:
                raise ValueError("Vector store dimension must be positive")
            
            # Validate security configuration
            if self.security.max_input_length <= 0:
                raise ValueError("Max input length must be positive")
            
            if self.security.max_file_size_mb <= 0:
                raise ValueError("Max file size must be positive")
            
            # Validate web configuration
            if not 1024 <= self.web.streamlit_port <= 65535:
                raise ValueError("Streamlit port must be between 1024 and 65535")
            
            if not 1024 <= self.web.fastapi_port <= 65535:
                raise ValueError("FastAPI port must be between 1024 and 65535")
            
            logger.info("Configuration validation passed")
            return True
            
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': {
                'name': self.model.name,
                'device': self.model.device,
                'max_length': self.model.max_length,
                'temperature': self.model.temperature,
                'top_p': self.model.top_p,
                'top_k': self.model.top_k,
                'do_sample': self.model.do_sample,
                'use_quantization': self.model.use_quantization,
                'quantization_bits': self.model.quantization_bits
            },
            'vector_store': {
                'type': self.vector_store.type,
                'dimension': self.vector_store.dimension,
                'similarity_threshold': self.vector_store.similarity_threshold,
                'max_results': self.vector_store.max_results,
                'persist_directory': self.vector_store.persist_directory
            },
            'embedding': {
                'model_name': self.embedding.model_name,
                'max_seq_length': self.embedding.max_seq_length,
                'batch_size': self.embedding.batch_size
            },
            'security': {
                'enable_content_filter': self.security.enable_content_filter,
                'enable_input_validation': self.security.enable_input_validation,
                'enable_rate_limiting': self.security.enable_rate_limiting,
                'max_input_length': self.security.max_input_length,
                'allowed_file_types': self.security.allowed_file_types,
                'max_file_size_mb': self.security.max_file_size_mb
            },
            'web': {
                'streamlit_port': self.web.streamlit_port,
                'fastapi_port': self.web.fastapi_port,
                'host': self.web.host,
                'debug': self.web.debug,
                'cors_origins': self.web.cors_origins
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_file_size_mb': self.logging.max_file_size_mb,
                'backup_count': self.logging.backup_count
            },
            'data_dir': self.data_dir,
            'documents_dir': self.documents_dir,
            'vector_store_dir': self.vector_store_dir,
            'logs_dir': self.logs_dir
        }


def get_config() -> Config:
    """Get application configuration."""
    # Try to load from config file first
    config_file = os.getenv('CONFIG_FILE', 'config/config.yaml')
    
    if os.path.exists(config_file):
        config = Config.from_file(config_file)
        logger.info(f"Loaded configuration from {config_file}")
    else:
        config = Config.from_env()
        logger.info("Loaded configuration from environment variables")
    
    # Validate and create directories
    if config.validate():
        config.create_directories()
    else:
        logger.error("Configuration validation failed. Using default configuration.")
        config = Config()
        config.create_directories()
    
    return config
