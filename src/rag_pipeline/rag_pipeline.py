"""
RAG Pipeline Module

Provides the core RAG (Retrieval-Augmented Generation) pipeline functionality.
Orchestrates the entire process from document ingestion to answer generation.
"""
import os
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import json

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator
from ..vector_stores.faiss_store import FAISSVectorStore
from ..vector_stores.chroma_store import ChromaVectorStore
from ..llm_integration.huggingface_llm import HuggingFaceLLM

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for document ingestion, retrieval, and generation."""
    
    def __init__(
        self,
        llm_model: str = 'mistral-7b',
        embedding_model: str = 'all-MiniLM-L6-v2',
        vector_store_type: str = 'chroma',  # 'chroma' or 'faiss'
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        data_dir: str = './data',
        **kwargs
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        self.data_dir = Path(data_dir)
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            device=kwargs.get('embedding_device', 'cpu')
        )
        
        # Initialize vector store
        self._initialize_vector_store(**kwargs)
        
        # Initialize LLM
        self.llm = HuggingFaceLLM(
            model_name=llm_model,
            device=kwargs.get('llm_device', 'auto'),
            load_in_4bit=kwargs.get('load_in_4bit', True),
            use_auth_token=kwargs.get('hf_token', None)
        )
        
        # Store for document metadata
        self.document_metadata = {}
        self._load_metadata()
    
    def _initialize_vector_store(self, **kwargs):
        """Initialize the vector store."""
        try:
            embedding_dim = self.embedding_generator.get_dimension()
            
            if self.vector_store_type == 'faiss':
                self.vector_store = FAISSVectorStore(
                    embedding_model=self.embedding_generator,
                    dimension=embedding_dim
                )
                self.vector_store_path = self.data_dir / 'faiss_index'
            elif self.vector_store_type == 'chroma':
                persist_dir = str(self.data_dir / 'chroma_db')
                self.vector_store = ChromaVectorStore(
                    embedding_model=self.embedding_generator,
                    persist_directory=persist_dir,
                    collection_name=kwargs.get('collection_name', 'documents')
                )
                self.vector_store_path = persist_dir
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
            
            logger.info(f"Initialized {self.vector_store_type} vector store")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _load_metadata(self):
        """Load document metadata from disk."""
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.document_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.document_metadata)} documents")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.document_metadata = {}
    
    def _save_metadata(self):
        """Save document metadata to disk."""
        metadata_path = self.data_dir / 'metadata.json'
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.document_metadata, f, indent=2)
            logger.info("Saved document metadata")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def ingest_documents(
        self, 
        file_paths: Optional[List[str]] = None,
        directory_path: Optional[str] = None,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Ingest documents into the RAG pipeline."""
        try:
            logger.info("Starting document ingestion")
            
            # Process documents
            if directory_path:
                documents = self.document_processor.process_directory(
                    directory_path, file_extensions
                )
            elif file_paths:
                documents = []
                for file_path in file_paths:
                    doc_data = self.document_processor.process_file(file_path)
                    documents.append(doc_data)
            else:
                raise ValueError("Either file_paths or directory_path must be provided")
            
            if not documents:
                logger.warning("No documents to ingest")
                return {'status': 'no_documents', 'count': 0}
            
            # Extract chunks and metadata for vector store
            all_chunks = []
            chunk_metadata = []
            
            for doc in documents:
                doc_id = doc['metadata']['filename']
                
                # Store document metadata
                self.document_metadata[doc_id] = doc['metadata']
                
                for chunk in doc['chunks']:
                    all_chunks.append(chunk['text'])
                    chunk_metadata.append({
                        'document_id': doc_id,
                        'chunk_id': chunk['id'],
                        'start_pos': chunk['start_pos'],
                        'end_pos': chunk['end_pos'],
                        'length': chunk['length'],
                        **doc['metadata']
                    })
            
            # Add to vector store
            self.vector_store.add_documents(all_chunks, chunk_metadata)
            
            # Save vector store and metadata
            if hasattr(self.vector_store, 'save') and self.vector_store_type == 'faiss':
                self.vector_store.save(str(self.vector_store_path))
            
            self._save_metadata()
            
            # Get statistics
            stats = self.document_processor.get_document_stats(documents)
            stats['vector_store_stats'] = self.vector_store.get_stats()
            
            logger.info(f"Successfully ingested {len(documents)} documents with {len(all_chunks)} chunks")
            return {
                'status': 'success',
                'documents_processed': len(documents),
                'chunks_created': len(all_chunks),
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise
    
    def query(
        self, 
        question: str, 
        top_k: int = 5,
        include_sources: bool = True,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Query the RAG pipeline with a question."""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=top_k)
            
            if not relevant_docs:
                logger.warning("No relevant documents found")
                return {
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': [],
                    'confidence': 'low'
                }
            
            # Prepare context
            context_parts = []
            sources = []
            
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"Source {i+1}: {doc['content']}")
                if include_sources:
                    sources.append({
                        'rank': doc['rank'],
                        'score': doc['score'],
                        'metadata': doc['metadata']
                    })
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = self._create_rag_prompt(question, context)
            
            # Generate answer
            answer = self.llm.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature
            )
            
            # Clean up answer
            answer = self._clean_answer(answer)
            
            # Determine confidence based on similarity scores
            avg_score = sum(doc['score'] for doc in relevant_docs) / len(relevant_docs)
            confidence = self._determine_confidence(avg_score)
            
            result = {
                'answer': answer,
                'confidence': confidence,
                'context_used': len(relevant_docs)
            }
            
            if include_sources:
                result['sources'] = sources
            
            logger.info("Successfully generated answer")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a well-structured RAG prompt."""
        return f"""Based on the following context information, please answer the question. If the context doesn't contain enough information to answer the question, please say so clearly.

Context Information:
{context}

Question: {question}

Answer: Please provide a detailed and accurate answer based on the context provided. If you're uncertain about any part of the answer, please mention that uncertainty."""
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format the generated answer."""
        # Remove common artifacts
        answer = answer.strip()
        
        # Remove repeated phrases
        lines = answer.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line
        
        return '\n'.join(cleaned_lines)
    
    def _determine_confidence(self, avg_score: float) -> str:
        """Determine confidence level based on similarity scores."""
        # Note: Lower scores are better for distance metrics
        if avg_score < 0.3:
            return 'high'
        elif avg_score < 0.6:
            return 'medium'
        else:
            return 'low'
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the RAG pipeline configuration."""
        return {
            'llm_model': self.llm_model,
            'embedding_model': self.embedding_model,
            'vector_store_type': self.vector_store_type,
            'data_directory': str(self.data_dir),
            'document_count': len(self.document_metadata),
            'llm_info': self.llm.get_model_info(),
            'embedding_info': self.embedding_generator.get_model_info(),
            'vector_store_stats': self.vector_store.get_stats()
        }
    
    def reset_pipeline(self) -> None:
        """Reset the pipeline by clearing all documents and vector store."""
        try:
            if hasattr(self.vector_store, 'delete_collection'):
                self.vector_store.delete_collection()
            
            self.document_metadata = {}
            self._save_metadata()
            
            logger.info("Pipeline reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting pipeline: {e}")
            raise
