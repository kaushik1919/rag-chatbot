# Installation Guide

This guide provides detailed instructions for setting up and running the RAG Chatbot in different environments.

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10/11
- **RAM**: 4GB (8GB recommended)
- **CPU**: 4 cores (8+ recommended)
- **Storage**: 10GB free space (50GB+ recommended)
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+

### Recommended Requirements
- **RAM**: 16GB+ for better performance
- **GPU**: CUDA-compatible GPU for faster inference
- **Storage**: SSD for better I/O performance
- **Network**: Stable internet connection for model downloads

## Installation Methods

### Method 1: Docker Deployment (Recommended)

This is the easiest and most reliable way to deploy the RAG Chatbot.

#### Step 1: Install Docker
**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS:**
```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop
```

**Windows:**
```powershell
# Install Docker Desktop from https://docker.com/products/docker-desktop
```

#### Step 2: Clone Repository
```bash
git clone <repository-url>
cd rag_chatbot
```

#### Step 3: Configure Environment
```bash
# Copy environment template
cp config/.env.template .env

# Edit configuration (optional)
nano config/config.yaml
```

#### Step 4: Deploy
**Linux/macOS:**
```bash
chmod +x docker/deploy.sh
./docker/deploy.sh start-prod
```

**Windows:**
```cmd
docker\deploy.bat start-prod
```

### Method 2: Local Python Installation

For development or custom setups.

#### Step 1: Install Python 3.10+
**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

**macOS:**
```bash
brew install python@3.10
```

**Windows:**
Download from https://python.org/downloads/

#### Step 2: Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Configure Environment
```bash
cp config/.env.template .env
# Edit .env with your configuration
```

#### Step 5: Run Services
**Terminal 1 (API):**
```bash
uvicorn src.web_interface.fastapi_app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 (Web Interface):**
```bash
streamlit run src/web_interface/streamlit_app.py --server.port 8501
```

### Method 3: Kubernetes Deployment

For production-scale deployments.

#### Prerequisites
- Kubernetes cluster (local or cloud)
- kubectl configured
- Helm (optional)

#### Step 1: Create Namespace
```bash
kubectl create namespace rag-chatbot
```

#### Step 2: Deploy Services
```bash
kubectl apply -f k8s/
```

## Environment Configuration

### Configuration Files

#### config/config.yaml
Main configuration file:
```yaml
# LLM Settings
llm_model: "mistral-7b"
embedding_model: "all-MiniLM-L6-v2"
vector_store_type: "chroma"

# Document Processing
chunk_size: 1000
chunk_overlap: 200

# Security Settings
content_filter_level: "moderate"
session_timeout_minutes: 60
```

#### .env File
Environment variables:
```bash
# Optional: Hugging Face token for private models
HF_TOKEN=your_token_here

# Environment
RAG_ENV=production
LOG_LEVEL=INFO

# Resource limits
MAX_MEMORY_GB=8
MAX_CPU_CORES=4
```

### Model Configuration

#### Supported LLM Models
- **mistral-7b**: Good balance of performance and quality
- **llama2-7b**: Meta's LLaMA-2 7B model
- **llama2-13b**: Larger, more capable model (requires more RAM)
- **falcon-7b**: TII's Falcon model
- **codellama-7b**: Specialized for code generation

#### Supported Embedding Models
- **all-MiniLM-L6-v2**: Fast, lightweight (384 dimensions)
- **all-mpnet-base-v2**: Higher quality (768 dimensions)
- **multi-qa-MiniLM-L6-cos-v1**: Optimized for Q&A tasks

#### Vector Store Options
- **ChromaDB**: Better for persistent storage and metadata
- **FAISS**: Faster similarity search, memory-based

## Verification

### Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check web interface
curl http://localhost:8501
```

### Service Status
```bash
# Docker deployment
./docker/deploy.sh status

# Local deployment
ps aux | grep -E "(uvicorn|streamlit)"
```

### Logs
```bash
# Docker logs
./docker/deploy.sh logs

# Local logs
tail -f logs/rag_chatbot.log
```

## First-Time Setup

### 1. Access the Application
Navigate to http://localhost:8501

### 2. Initial Login
- Default admin user will be created automatically
- Check logs for the generated password
- Login and change the default password immediately

### 3. Upload Test Documents
- Use the sidebar to upload sample documents
- Supported formats: PDF, DOCX, TXT
- Wait for processing to complete

### 4. Test Chat Functionality
- Ask questions about your uploaded documents
- Verify responses include source citations
- Test different model configurations

## Troubleshooting

### Common Issues

#### "Docker not found" Error
```bash
# Verify Docker installation
docker --version
docker-compose --version

# Start Docker service (Linux)
sudo systemctl start docker
```

#### "Port already in use" Error
```bash
# Check what's using the port
netstat -tulpn | grep :8000

# Kill the process or change ports in docker-compose.yml
```

#### "Out of memory" Error
```bash
# Check available memory
free -h

# Enable swap
sudo swapon --show

# Use 4-bit quantization in config.yaml
load_in_4bit: true
```

#### Model Download Issues
```bash
# Check internet connection
ping huggingface.co

# Set Hugging Face cache directory
export HF_HOME=/path/to/large/storage

# Manually download models (optional)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')"
```

### Performance Issues

#### Slow Response Times
1. Enable 4-bit quantization
2. Use smaller models (7B instead of 13B)
3. Increase system RAM
4. Use GPU acceleration if available

#### High Memory Usage
1. Enable quantization
2. Reduce chunk_size in config
3. Use FAISS instead of ChromaDB
4. Limit concurrent requests

### Security Considerations

#### Production Deployment
1. Change default passwords
2. Configure SSL/TLS certificates
3. Set up firewall rules
4. Enable audit logging
5. Regular security updates

#### Network Security
```bash
# Restrict access to specific IPs
iptables -A INPUT -p tcp --dport 8000 -s YOUR_IP -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

## GPU Support

### NVIDIA GPU Setup

#### Install NVIDIA Docker
```bash
# Add NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

#### Enable GPU in Docker Compose
```yaml
services:
  rag-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Scaling and Production

### Horizontal Scaling
- Deploy multiple API instances behind a load balancer
- Use Redis for session storage
- Implement database replication for ChromaDB

### Monitoring
- Set up Prometheus and Grafana
- Configure log aggregation (ELK stack)
- Implement health check monitoring

### Backup Strategy
```bash
# Automated backup script
./docker/deploy.sh backup

# Schedule with cron
0 2 * * * /path/to/rag_chatbot/docker/deploy.sh backup
```

## Support

If you encounter issues during installation:

1. Check the [troubleshooting section](#troubleshooting)
2. Review application logs
3. Verify system requirements
4. Check Docker and network connectivity
5. Open an issue on GitHub with detailed error logs

For additional help, provide:
- Operating system and version
- Docker version
- Error messages and logs
- System specifications
- Installation method used
