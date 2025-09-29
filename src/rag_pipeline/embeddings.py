"""
Embeddings generator using sentence transformers.
"""
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence transformers."""
    
    # Supported embedding models
    SUPPORTED_MODELS = {
        'all-MiniLM-L6-v2': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'max_seq_length': 256,
            'description': 'Fast and efficient, good for general use'
        },
        'all-mpnet-base-v2': {
            'model_name': 'sentence-transformers/all-mpnet-base-v2',
            'dimension': 768,
            'max_seq_length': 384,
            'description': 'High quality embeddings, slower but more accurate'
        },
        'multi-qa-MiniLM-L6-cos-v1': {
            'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'dimension': 384,
            'max_seq_length': 512,
            'description': 'Optimized for question-answering tasks'
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'dimension': 384,
            'max_seq_length': 128,
            'description': 'Multilingual support'
        }
    }
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_config = self.SUPPORTED_MODELS[model_name]
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_config['model_name']}")
            
            self.model = SentenceTransformer(
                self.model_config['model_name'],
                device=self.device
            )
            
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            if not self.model:
                raise RuntimeError("Model not loaded")
            
            if not texts:
                return np.array([])
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.encode([text])[0]
    
    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings."""
        try:
            # Ensure embeddings are normalized
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarity = np.dot(embeddings1, embeddings2.T)
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        document_embeddings: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find most similar documents to a query."""
        try:
            if len(document_embeddings) == 0:
                return []
            
            # Compute similarities
            similarities = self.compute_similarity(
                query_embedding.reshape(1, -1), 
                document_embeddings
            )[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Format results
            results = []
            for i, idx in enumerate(top_indices):
                results.append({
                    'index': int(idx),
                    'similarity': float(similarities[idx]),
                    'rank': i + 1
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            'model_name': self.model_name,
            'model_path': self.model_config['model_name'],
            'dimension': self.model_config['dimension'],
            'max_seq_length': self.model_config['max_seq_length'],
            'description': self.model_config['description'],
            'device': self.device,
            'is_loaded': self.model is not None
        }
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model_config['dimension']
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """Get list of available embedding models."""
        return EmbeddingGenerator.SUPPORTED_MODELS
