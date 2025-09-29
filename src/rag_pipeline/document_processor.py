"""
Document processing utilities for RAG pipeline.
"""
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from PyPDF2 import PdfReader
import docx

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process various document formats for RAG pipeline."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and return document data."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type and extract content
            content = ""
            file_type = file_path.suffix.lower()
            
            if file_type == '.pdf':
                content = self._extract_pdf_content(file_path)
            elif file_type == '.docx':
                content = self._extract_docx_content(file_path)
            elif file_type == '.txt':
                content = self._extract_text_content(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Create chunks
            chunks = self._create_chunks(content)
            
            # Create metadata
            metadata = {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_type': file_type,
                'file_size': file_path.stat().st_size,
                'num_chunks': len(chunks)
            }
            
            return {
                'content': content,
                'chunks': chunks,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF."""
        try:
            reader = PdfReader(str(file_path))
            content = ""
            
            for page in reader.pages:
                content += page.extract_text() + "\n"
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise
    
    def _extract_docx_content(self, file_path: Path) -> str:
        """Extract text content from DOCX."""
        try:
            doc = docx.Document(str(file_path))
            content = ""
            
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting DOCX content: {e}")
            raise
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
    
    def _create_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Split content into chunks with overlap."""
        try:
            # Clean content
            content = self._clean_content(content)
            
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < len(content):
                # Calculate end position
                end = start + self.chunk_size
                
                # Adjust end to avoid breaking words
                if end < len(content):
                    # Find the last space before the end
                    last_space = content.rfind(' ', start, end)
                    if last_space > start:
                        end = last_space
                
                # Extract chunk
                chunk_text = content[start:end].strip()
                
                if chunk_text:
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'start_pos': start,
                        'end_pos': end,
                        'length': len(chunk_text)
                    })
                    chunk_id += 1
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                
                # Prevent infinite loop
                if start >= end:
                    start = end
            
            logger.info(f"Created {len(chunks)} chunks from content")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might interfere
        content = re.sub(r'[^\w\s\.,!?;:()\-\'""]', ' ', content)
        
        # Remove excessive line breaks
        content = re.sub(r'\n+', '\n', content)
        
        return content.strip()
    
    def process_directory(self, directory_path: str, file_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process all supported files in a directory."""
        if file_extensions is None:
            file_extensions = ['.pdf', '.docx', '.txt']
        
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        processed_files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    doc_data = self.process_file(str(file_path))
                    processed_files.append(doc_data)
                    logger.info(f"Processed: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
        
        logger.info(f"Processed {len(processed_files)} files from {directory}")
        return processed_files
    
    def get_document_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        total_docs = len(documents)
        total_chunks = sum(doc['metadata']['num_chunks'] for doc in documents)
        total_size = sum(doc['metadata']['file_size'] for doc in documents)
        
        file_types = {}
        for doc in documents:
            file_type = doc['metadata']['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'total_size_bytes': total_size,
            'file_types': file_types,
            'avg_chunks_per_doc': total_chunks / total_docs if total_docs > 0 else 0
        }
