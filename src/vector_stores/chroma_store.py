"""
Chroma vector store implementation for persistent document storage.
"""
import chromadb
import os
from typing import List, Dict, Any, Optional
import logging
from .base import VectorStoreBase

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB-based vector store with persistence."""
    
    def __init__(self, embedding_model: Any, persist_directory: str = "./data/chroma_db", 
                 collection_name: str = "documents", **kwargs):
        super().__init__(embedding_model, **kwargs)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Add documents to ChromaDB collection."""
        try:
            logger.info(f"Adding {len(documents)} documents to Chroma collection")
            
            # Generate embeddings
            embeddings = self.get_embeddings(documents)
            
            # Prepare IDs
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
            
            # Prepare metadata
            if metadatas is None:
                metadatas = [{"source": "unknown"} for _ in documents]
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to Chroma")
            
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents using ChromaDB."""
        try:
            # Generate query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count())
            )
            
            # Format results
            formatted_results = []
            if results['documents'][0]:  # Check if results exist
                for i, (doc, metadata, distance) in enumerate(
                    zip(results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0])
                ):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': float(distance),
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching Chroma collection: {e}")
            raise
    
    def save(self, path: str) -> None:
        """ChromaDB automatically persists data."""
        logger.info("ChromaDB data is automatically persisted")
    
    def load(self, path: str) -> None:
        """ChromaDB automatically loads persisted data."""
        logger.info("ChromaDB data is automatically loaded from persist directory")
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'total_documents': 0}
    
    def update_document(self, doc_id: str, document: str, metadata: Optional[Dict] = None) -> None:
        """Update an existing document."""
        try:
            embedding = self.get_embeddings([document])[0]
            
            self.collection.update(
                ids=[doc_id],
                documents=[document],
                embeddings=[embedding],
                metadatas=[metadata] if metadata else None
            )
            
            logger.info(f"Updated document: {doc_id}")
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document by ID."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
