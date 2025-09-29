"""
FAISS vector store implementation for efficient similarity search.
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional
import logging
from .base import VectorStoreBase

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStoreBase):
    """FAISS-based vector store for fast similarity search."""
    
    def __init__(self, embedding_model: Any, dimension: int = 384, **kwargs):
        super().__init__(embedding_model, **kwargs)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadatas = []
        
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Add documents to the FAISS index."""
        try:
            logger.info(f"Adding {len(documents)} documents to FAISS index")
            
            # Generate embeddings
            embeddings = self.get_embeddings(documents)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store documents and metadata
            self.documents.extend(documents)
            if metadatas:
                self.metadatas.extend(metadatas)
            else:
                self.metadatas.extend([{} for _ in documents])
                
            logger.info(f"Successfully added {len(documents)} documents. Total: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents using FAISS."""
        try:
            if len(self.documents) == 0:
                logger.warning("No documents in index")
                return []
            
            # Generate query embedding
            query_embedding = self.get_embeddings([query])[0]
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Search
            k = min(k, len(self.documents))
            distances, indices = self.index.search(query_array, k)
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    results.append({
                        'content': self.documents[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(distance),
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            raise
    
    def save(self, path: str) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.faiss")
            
            # Save documents and metadata
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"Saved FAISS index to {path}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise
    
    def load(self, path: str) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.faiss")
            
            # Load documents and metadata
            with open(f"{path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']
                self.dimension = data['dimension']
            
            logger.info(f"Loaded FAISS index from {path} with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_documents': len(self.documents),
            'dimension': self.dimension,
            'index_size': self.index.ntotal if self.index else 0
        }
