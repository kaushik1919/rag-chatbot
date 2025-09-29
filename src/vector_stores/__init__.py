"""
Vector store implementations for document retrieval.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """Abstract interface for vector store implementations."""
    
    def __init__(self, embedding_model: Any, **kwargs):
        self.embedding_model = embedding_model
        self.index = None
        
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts."""
        try:
            if hasattr(self.embedding_model, 'encode'):
                return self.embedding_model.encode(texts).tolist()
            else:
                return [self.embedding_model.embed_query(text) for text in texts]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
