#!/bin/bash
# Docker deployment and management scripts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    log_info "Docker is running"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p data/chroma_db
    mkdir -p data/faiss_index
    mkdir -p logs
    mkdir -p config
    chmod 755 data logs config
    log_info "Directories created"
}

# Build and start services
start_production() {
    log_info "Starting RAG Chatbot in production mode..."
    check_docker
    create_directories
    
    # Build and start services
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    
    log_info "Services starting up..."
    sleep 10
    
    # Check service health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✅ API service is healthy"
    else
        log_error "❌ API service health check failed"
    fi
    
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        log_info "✅ Web interface is accessible"
    else
        log_warn "⚠️ Web interface may still be starting up"
    fi
    
    log_info "RAG Chatbot started successfully!"
    log_info "API: http://localhost:8000"
    log_info "Web Interface: http://localhost:8501"
    log_info "API Docs: http://localhost:8000/docs"
}

# Start development environment
start_development() {
    log_info "Starting RAG Chatbot in development mode..."
    check_docker
    create_directories
    
    docker-compose -f docker-compose.dev.yml down
    docker-compose -f docker-compose.dev.yml build --no-cache
    docker-compose -f docker-compose.dev.yml up -d
    
    log_info "Development environment started!"
    log_info "API (with hot reload): http://localhost:8000"
    log_info "Web Interface (with hot reload): http://localhost:8501"
}

# Stop services
stop_services() {
    log_info "Stopping RAG Chatbot services..."
    docker-compose down
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    log_info "Services stopped"
}

# View logs
view_logs() {
    service=${1:-""}
    if [ -n "$service" ]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

# Clean up Docker resources
cleanup() {
    log_warn "This will remove all RAG Chatbot containers, images, and volumes"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning up Docker resources..."
        docker-compose down -v --rmi all --remove-orphans
        docker system prune -f
        log_info "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show service status
status() {
    log_info "RAG Chatbot Service Status:"
    docker-compose ps
    
    echo
    log_info "Service Health:"
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "API: ${GREEN}✅ Healthy${NC}"
    else
        echo -e "API: ${RED}❌ Unhealthy${NC}"
    fi
    
    # Check Web interface
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        echo -e "Web: ${GREEN}✅ Accessible${NC}"
    else
        echo -e "Web: ${RED}❌ Inaccessible${NC}"
    fi
}

# Update services
update() {
    log_info "Updating RAG Chatbot services..."
    git pull
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    log_info "Update completed"
}

# Backup data
backup() {
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    log_info "Creating backup in $backup_dir..."
    
    mkdir -p "$backup_dir"
    
    # Backup data directory
    if [ -d "data" ]; then
        cp -r data "$backup_dir/"
        log_info "Data backed up"
    fi
    
    # Backup configuration
    if [ -d "config" ]; then
        cp -r config "$backup_dir/"
        log_info "Configuration backed up"
    fi
    
    # Create archive
    tar -czf "$backup_dir.tar.gz" "$backup_dir"
    rm -rf "$backup_dir"
    
    log_info "Backup created: $backup_dir.tar.gz"
}

# Install system requirements (Ubuntu/Debian)
install_requirements() {
    log_info "Installing system requirements..."
    
    # Update package list
    sudo apt-get update
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        log_info "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        log_info "Docker installed. Please logout and login again."
    fi
    
    # Install Docker Compose if not present
    if ! command -v docker-compose &> /dev/null; then
        log_info "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        log_info "Docker Compose installed"
    fi
    
    log_info "System requirements installed"
}

# Show help
show_help() {
    echo "RAG Chatbot Docker Management Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start-prod          Start in production mode"
    echo "  start-dev           Start in development mode"
    echo "  stop               Stop all services"
    echo "  restart            Restart all services"
    echo "  status             Show service status"
    echo "  logs [service]     View logs (optionally for specific service)"
    echo "  update             Update services"
    echo "  backup             Create backup of data and config"
    echo "  cleanup            Remove all Docker resources"
    echo "  install-req        Install system requirements"
    echo "  help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0 start-prod      # Start production environment"
    echo "  $0 logs rag-api    # View API logs"
    echo "  $0 backup          # Create backup"
}

# Main script logic
case "${1:-help}" in
    "start-prod"|"prod")
        start_production
        ;;
    "start-dev"|"dev")
        start_development
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_production
        ;;
    "status")
        status
        ;;
    "logs")
        view_logs "$2"
        ;;
    "update")
        update
        ;;
    "backup")
        backup
        ;;
    "cleanup")
        cleanup
        ;;
    "install-req")
        install_requirements
        ;;
    "help"|*)
        show_help
        ;;
esac
