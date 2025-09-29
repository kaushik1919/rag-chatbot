# 🤖 RAG Chatbot - Retrieval-Augmented Generation with LangChain & LlamaIndex

A comprehensive, production-ready RAG (Retrieval-Augmented Generation) chatbot built with open-source LLMs, featuring advanced security controls, multiple vector databases, and containerized deployment.

## ✨ Features

### 🧠 **Advanced RAG Pipeline**
- **Open-Source LLMs**: Support for LLaMA-2, Mistral, Falcon, and CodeLlama models
- **Multiple Vector Stores**: FAISS and ChromaDB integration for efficient similarity search
- **Smart Document Processing**: PDF, DOCX, and TXT support with intelligent chunking
- **Sophisticated Embeddings**: Sentence Transformers with multiple model options
- **Prompt Engineering**: Advanced templates and chain-of-thought reasoning

### 🔒 **Enterprise Security**
- **Input Validation**: Comprehensive sanitization and security checks
- **Content Filtering**: Multi-level safety controls with configurable strictness
- **Access Control**: Role-based permissions (Admin, User, ReadOnly, Guest)
- **Session Management**: Secure authentication with session timeout
- **Rate Limiting**: Configurable API and query rate limits

### 🌐 **Dual Interface**
- **Streamlit Web App**: User-friendly chat interface with file upload
- **FastAPI Backend**: RESTful API with comprehensive documentation
- **Real-time Chat**: Interactive conversations with source citations
- **Document Management**: Upload, process, and manage knowledge base

### 🚀 **Production Ready**
- **Docker Deployment**: Complete containerization with docker-compose
- **Nginx Reverse Proxy**: Load balancing and SSL termination
- **Health Monitoring**: Comprehensive health checks and metrics
- **Logging**: Structured logging with rotation and JSON output
- **Configuration Management**: YAML-based configuration with environment variables

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │    Vector DB    │
│  Web Interface  │◄──►│   REST API      │◄──►│ (FAISS/Chroma)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Core                             │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Document       │   Embeddings    │      LLM Integration        │
│  Processor      │   Generator     │   (HuggingFace Models)      │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Governance & Security Layer                     │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Input Validation│ Content Filter  │    Access Control           │
│   & Sanitization│  & Safety       │   & Authentication          │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+ (for local development)
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster inference)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag_chatbot
```

### 2. Configure Environment
```bash
# Copy environment template
cp config/.env.template .env

# Edit configuration (optional)
nano config/config.yaml
```

### 3. Deploy with Docker

#### Production Deployment
```bash
# Linux/Mac
chmod +x docker/deploy.sh
./docker/deploy.sh start-prod

# Windows
docker\deploy.bat start-prod
```

#### Development Deployment
```bash
# Linux/Mac
./docker/deploy.sh start-dev

# Windows
docker\deploy.bat start-dev
```

### 4. Access the Application
- **Web Interface**: http://localhost:8501
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 5. First Login
- Default admin credentials will be displayed in logs
- Change the default password immediately via the web interface

## 📚 Usage Guide

### Document Upload
1. Access the web interface at http://localhost:8501
2. Login with your credentials
3. Use the sidebar to upload PDF, DOCX, or TXT files
4. Wait for processing to complete

### Chatting with Documents
1. Select your preferred model configuration
2. Type questions in the chat interface
3. View responses with source citations
4. Adjust parameters as needed

### API Usage
```python
import requests

# Login
response = requests.post("http://localhost:8000/auth/login", 
                        json={"username": "admin", "password": "your_password"})
token = response.json()["data"]["session_token"]

# Query the chatbot
headers = {"Authorization": f"Bearer {token}"}
query = {"question": "What is the main topic of the uploaded documents?"}
response = requests.post("http://localhost:8000/query", 
                        json=query, headers=headers)
print(response.json()["answer"])
```

## 🔧 Configuration

### Model Configuration
Edit `config/config.yaml`:
```yaml
llm_model: "mistral-7b"  # Options: mistral-7b, llama2-7b, falcon-7b
embedding_model: "all-MiniLM-L6-v2"
vector_store_type: "chroma"  # Options: chroma, faiss
```

### Security Settings
```yaml
content_filter_level: "moderate"  # strict, moderate, relaxed
session_timeout_minutes: 60
max_login_attempts: 5
```

### Performance Tuning
```yaml
chunk_size: 1000
chunk_overlap: 200
temperature: 0.7
top_k_retrieval: 5
```

## 🛠️ Development

### Local Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run components locally
# API
uvicorn src.web_interface.fastapi_app:app --reload

# Web Interface
streamlit run src/web_interface/streamlit_app.py
```

### Project Structure
```
rag_chatbot/
├── src/
│   ├── vector_stores/          # FAISS & ChromaDB implementations
│   ├── llm_integration/        # HuggingFace LLM wrappers
│   ├── rag_pipeline/           # Core RAG pipeline
│   ├── governance/             # Security & access control
│   └── web_interface/          # Streamlit & FastAPI apps
├── config/                     # Configuration files
├── docker/                     # Docker & deployment scripts
├── data/                       # Persistent data storage
├── logs/                       # Application logs
└── tests/                      # Test suite
```

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/

# Lint code
flake8 src/
black src/
```

## 🔒 Security Features

### Input Validation
- SQL injection prevention
- Command injection protection
- Path traversal detection
- Content sanitization

### Content Filtering
- Hate speech detection
- Violence content filtering
- Personal information redaction
- Spam prevention

### Access Control
- Role-based permissions
- Session management
- Rate limiting
- Failed login protection

### Data Governance
- Audit logging
- Content filtering logs
- Access attempt tracking
- Data retention policies

## 📊 Monitoring & Observability

### Health Checks
```bash
# Check service health
curl http://localhost:8000/health

# View service status
./docker/deploy.sh status
```

### Logs
```bash
# View all logs
./docker/deploy.sh logs

# View specific service logs
./docker/deploy.sh logs rag-api
```

### Metrics
- Response times
- Query success rates
- Document processing statistics
- User activity metrics

## 🐳 Docker Management

### Common Commands
```bash
# Start production environment
./docker/deploy.sh start-prod

# Start development environment
./docker/deploy.sh start-dev

# Stop all services
./docker/deploy.sh stop

# View logs
./docker/deploy.sh logs

# Check status
./docker/deploy.sh status

# Update services
./docker/deploy.sh update

# Backup data
./docker/deploy.sh backup

# Clean up resources
./docker/deploy.sh cleanup
```

## 📈 Performance Optimization

### Model Optimization
- **4-bit Quantization**: Reduces memory usage by ~75%
- **Model Caching**: Faster subsequent loads
- **Batch Processing**: Efficient document ingestion

### Vector Store Optimization
- **FAISS**: Faster similarity search for large datasets
- **ChromaDB**: Better for persistent storage and metadata
- **Index Optimization**: Configurable index types

### System Requirements
| Component | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4GB | 16GB+ |
| CPU | 4 cores | 8+ cores |
| Storage | 10GB | 50GB+ SSD |
| GPU | None | CUDA-compatible |

## 🚨 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check available memory
docker stats

# Reduce model size
# Edit config/config.yaml - set load_in_4bit: true
```

#### Connection Errors
```bash
# Check service status
./docker/deploy.sh status

# Restart services
./docker/deploy.sh stop
./docker/deploy.sh start-prod
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check logs for bottlenecks
./docker/deploy.sh logs rag-api
```

### Debug Mode
```bash
# Enable debug logging
# Edit config/config.yaml - set log_level: "DEBUG"

# View detailed logs
./docker/deploy.sh logs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guide
- Add type hints
- Write comprehensive tests
- Update documentation
- Use semantic commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HuggingFace** for transformers and model hosting
- **LangChain** for RAG framework inspiration
- **ChromaDB** for vector database technology
- **FAISS** for efficient similarity search
- **Streamlit** for rapid web app development
- **FastAPI** for modern API framework

## 📞 Support

For support and questions:
- Check the [troubleshooting guide](#-troubleshooting)
- Review [logs](#logs) for error details
- Open an issue on GitHub
- Check the [API documentation](http://localhost:8000/docs)

## 🗺️ Roadmap

### Upcoming Features
- [ ] Multi-modal support (images, audio)
- [ ] Advanced RAG techniques (HyDE, RAPTOR)
- [ ] Integration with more LLM providers
- [ ] Enhanced security features
- [ ] Performance monitoring dashboard
- [ ] Multi-language support
- [ ] Advanced analytics and reporting

---


