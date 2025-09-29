# Contributing to RAG Chatbot

Thank you for your interest in contributing to the RAG Chatbot project! This document provides guidelines for contributing to make the process smooth and effective for everyone.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Process](#contribution-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a code of conduct that we expect all participants to follow. Please be respectful, inclusive, and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git
- Basic understanding of RAG (Retrieval-Augmented Generation) concepts

### Areas for Contribution

We welcome contributions in the following areas:

- **Core Features**: RAG pipeline improvements, new model integrations
- **Security**: Enhanced governance controls, vulnerability fixes
- **Performance**: Optimization, caching, efficiency improvements
- **Documentation**: User guides, API documentation, examples
- **Testing**: Unit tests, integration tests, performance tests
- **DevOps**: CI/CD, deployment improvements, monitoring
- **UI/UX**: Web interface enhancements, accessibility improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### 2. Set Up Development Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### 3. Configure Environment

```bash
# Copy environment template
cp config/.env.template .env

# Edit configuration for development
# Set RAG_ENV=development in .env
```

### 4. Run Development Environment

```bash
# Option 1: Docker (recommended)
./docker/deploy.sh start-dev

# Option 2: Local development
# Terminal 1 - API
uvicorn src.web_interface.fastapi_app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Web Interface  
streamlit run src/web_interface/streamlit_app.py --server.port 8501
```

## Contribution Process

### 1. Create an Issue

Before starting work, create an issue to discuss your proposed changes:

- **Bug Reports**: Use the bug report template
- **Feature Requests**: Use the feature request template
- **Documentation**: Describe what needs improvement

### 2. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b bugfix/issue-number-description
```

### 3. Make Changes

Follow these guidelines when making changes:

- Keep commits small and focused
- Write clear commit messages
- Update documentation as needed
- Add tests for new functionality

### 4. Test Your Changes

```bash
# Run tests
python -m pytest tests/ -v

# Run linting
flake8 src/ tests/
black --check src/ tests/

# Run type checking (if mypy is installed)
mypy src/

# Test Docker build
docker build -t rag-chatbot-test .
```

### 5. Submit Pull Request

1. Push your branch to your fork
2. Create a pull request against the main branch
3. Fill out the pull request template
4. Link to related issues
5. Request review from maintainers

## Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def process_document(file_path: str) -> Dict[str, Any]:
    """Process a document and return metadata."""
    pass

# Use descriptive variable names
embedding_dimension = 384  # Good
dim = 384  # Avoid

# Document classes and functions
class DocumentProcessor:
    """Processes documents for RAG pipeline ingestion."""
    
    def __init__(self, chunk_size: int = 1000):
        """Initialize processor with specified chunk size."""
        self.chunk_size = chunk_size
```

### Code Organization

```
src/
├── vector_stores/     # Vector database implementations
├── llm_integration/   # Language model wrappers
├── rag_pipeline/      # Core RAG logic
├── governance/        # Security and access control
└── web_interface/     # User interfaces
```

### Import Organization

```python
# Standard library imports
import os
import logging
from typing import Dict, List, Optional

# Third-party imports
import torch
from transformers import AutoModel

# Local imports
from .base import BaseClass
from ..utils import helper_function
```

### Error Handling

```python
# Use specific exceptions
try:
    result = process_data(input_data)
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error in process_data: {e}")
    raise ProcessingError(f"Failed to process data: {e}")
```

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests
├── end_to_end/       # Full system tests
├── fixtures/         # Test data and fixtures
└── conftest.py       # Pytest configuration
```

### Writing Tests

```python
import pytest
from src.rag_pipeline.document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance for testing."""
        return DocumentProcessor(chunk_size=100)
    
    def test_process_text_file(self, processor, tmp_path):
        """Test processing of text files."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Process file
        result = processor.process_file(str(test_file))
        
        # Assertions
        assert result['metadata']['filename'] == 'test.txt'
        assert len(result['chunks']) > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_document_processor.py

# Run with coverage
pytest --cov=src/ --cov-report=html

# Run integration tests
pytest tests/integration/
```

## Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Include parameter types and return types
- Provide usage examples for complex functions

```python
def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for documents similar to the query.
    
    Args:
        query: The search query text
        k: Number of similar documents to return (default: 5)
        
    Returns:
        List of dictionaries containing document content and metadata
        
    Example:
        >>> results = store.similarity_search("machine learning", k=3)
        >>> print(f"Found {len(results)} similar documents")
    """
```

### API Documentation

- Update OpenAPI schemas when changing API endpoints
- Include request/response examples
- Document error responses

### User Documentation

- Update README.md for user-facing changes
- Update INSTALL.md for setup changes
- Add usage examples to docs/

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment Information**:
   - Operating system and version
   - Python version
   - Docker version (if applicable)
   - RAG Chatbot version

2. **Steps to Reproduce**:
   - Exact steps to reproduce the issue
   - Expected behavior
   - Actual behavior

3. **Logs and Error Messages**:
   - Complete error messages
   - Relevant log entries
   - Screenshots (if applicable)

4. **Additional Context**:
   - Configuration settings
   - Model versions used
   - Data characteristics

### Security Issues

For security vulnerabilities:

1. **Do NOT open a public issue**
2. Email the maintainers directly
3. Include detailed reproduction steps
4. Provide suggested fixes if possible

## Feature Requests

When requesting features:

1. **Describe the Use Case**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives Considered**: What other approaches did you consider?
4. **Implementation Ideas**: Any thoughts on how to implement it?

## Performance Considerations

When contributing performance improvements:

1. **Benchmark Before and After**: Provide performance metrics
2. **Profile Critical Paths**: Use profiling tools to identify bottlenecks
3. **Consider Memory Usage**: Balance speed with memory consumption
4. **Test with Different Data Sizes**: Ensure scalability

## Model Integration

When adding new model support:

1. **Follow Existing Patterns**: Use the same interface as existing models
2. **Add Configuration Options**: Make models configurable
3. **Document Requirements**: List hardware/memory requirements
4. **Provide Examples**: Show how to use the new model

## UI/UX Contributions

For interface improvements:

1. **Maintain Accessibility**: Follow WCAG guidelines
2. **Test Across Browsers**: Ensure compatibility
3. **Consider Mobile Users**: Test responsive design
4. **Follow Design Patterns**: Maintain consistency

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Getting Help

If you need help:

1. Check existing documentation
2. Search existing issues
3. Ask in discussions
4. Join our community chat (if available)
5. Tag maintainers in issues/PRs

## Recognition

Contributors are recognized in:

- AUTHORS.md file
- Release notes
- GitHub contributors page
- Special mentions for significant contributions

Thank you for contributing to RAG Chatbot! Your efforts help make this project better for everyone.
