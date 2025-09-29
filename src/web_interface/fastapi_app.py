"""
FastAPI backend for the RAG chatbot.
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import tempfile
import os
import logging
from pathlib import Path
import time
from datetime import datetime

# Import RAG components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rag_pipeline.rag_pipeline import RAGPipeline
from rag_pipeline.prompt_engineering import PromptEngineer, PromptType
from governance.input_validation import InputValidator
from governance.content_filter import ContentFilter, FilterLevel
from governance.access_control import AccessController, UserRole, Permission

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="REST API for Retrieval-Augmented Generation Chatbot with governance controls",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global components
rag_pipeline = None
input_validator = InputValidator()
content_filter = ContentFilter(FilterLevel.MODERATE)
access_controller = AccessController()
prompt_engineer = PromptEngineer()


# Pydantic models
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=100)


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    role: str = Field(..., regex="^(admin|user|readonly|guest)$")
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    include_sources: Optional[bool] = True
    max_length: Optional[int] = Field(None, ge=100, le=4000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class ConfigRequest(BaseModel):
    llm_model: Optional[str] = Field("mistral-7b", regex="^(mistral-7b|llama2-7b|llama2-13b|falcon-7b|codellama-7b)$")
    embedding_model: Optional[str] = Field("all-MiniLM-L6-v2", regex="^(all-MiniLM-L6-v2|all-mpnet-base-v2|multi-qa-MiniLM-L6-cos-v1)$")
    vector_store_type: Optional[str] = Field("chroma", regex="^(chroma|faiss)$")
    chunk_size: Optional[int] = Field(1000, ge=100, le=5000)
    chunk_overlap: Optional[int] = Field(200, ge=0, le=500)


class DocumentIngestRequest(BaseModel):
    file_paths: Optional[List[str]] = None
    directory_path: Optional[str] = None
    file_extensions: Optional[List[str]] = ['.pdf', '.txt', '.docx']


# Response models
class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class QueryResponse(BaseModel):
    answer: str
    confidence: str
    context_used: int
    sources: Optional[List[Dict[str, Any]]] = None
    response_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class SystemInfoResponse(BaseModel):
    pipeline_info: Dict[str, Any]
    system_health: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Validate authentication token and return user info."""
    try:
        session_info = access_controller.validate_session(credentials.credentials)
        if not session_info.get('valid', False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session token"
            )
        return session_info
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


# Permission check dependency
def require_permission(permission: Permission):
    """Create dependency that requires specific permission."""
    def permission_dependency(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = UserRole(user['role'])
        user_permissions = access_controller.role_permissions.get(user_role, set())
        
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission.value}"
            )
        return user
    
    return permission_dependency


# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now(),
        "endpoints": {
            "auth": "/auth/*",
            "query": "/query",
            "documents": "/documents/*",
            "system": "/system/*",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "api": True,
        "pipeline": rag_pipeline is not None,
        "governance": {
            "input_validator": input_validator is not None,
            "content_filter": content_filter is not None,
            "access_controller": access_controller is not None
        },
        "timestamp": datetime.now()
    }
    
    all_healthy = all([
        health_status["api"],
        health_status["pipeline"],
        all(health_status["governance"].values())
    ])
    
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content=health_status
    )


# Authentication endpoints
@app.post("/auth/login", response_model=StandardResponse)
async def login(request: LoginRequest):
    """Authenticate user and return session token."""
    try:
        # Validate input
        validation_result = input_validator.validate_query(request.username)
        if not validation_result['is_valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid username format"
            )
        
        # Authenticate
        auth_result = access_controller.authenticate_user(request.username, request.password)
        
        if auth_result['success']:
            return StandardResponse(
                success=True,
                message="Login successful",
                data={
                    "session_token": auth_result['session_token'],
                    "user_info": auth_result['user_info']
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=auth_result['error']
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/auth/logout", response_model=StandardResponse)
async def logout(user: Dict[str, Any] = Depends(get_current_user)):
    """Logout user by invalidating session."""
    try:
        # Session token is in the authorization header
        success = access_controller.logout_user(user.get('session_token', ''))
        
        return StandardResponse(
            success=success,
            message="Logout successful" if success else "Logout failed"
        )
    
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/auth/create-user", response_model=StandardResponse)
async def create_user(
    request: UserCreateRequest,
    user: Dict[str, Any] = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Create new user (admin only)."""
    try:
        result = access_controller.create_user(
            username=request.username,
            password=request.password,
            role=UserRole(request.role),
            email=request.email
        )
        
        if result['success']:
            return StandardResponse(
                success=True,
                message="User created successfully",
                data={"username": result['username'], "role": result['role']}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Query endpoints
@app.post("/query", response_model=QueryResponse)
async def query_chatbot(
    request: QueryRequest,
    user: Dict[str, Any] = Depends(require_permission(Permission.QUERY_SYSTEM))
):
    """Query the RAG chatbot."""
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )
        
        # Validate input
        validation_result = input_validator.validate_query(request.question)
        if not validation_result['is_valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid query: {validation_result['errors']}"
            )
        
        # Safety check
        safety_result = content_filter.check_query_safety(request.question)
        if not safety_result['should_process']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query blocked by content filter: {safety_result['warnings']}"
            )
        
        # Process query
        start_time = time.time()
        
        response_data = rag_pipeline.query(
            question=validation_result['sanitized_query'],
            top_k=request.top_k,
            include_sources=request.include_sources,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        end_time = time.time()
        
        # Filter response
        filtered_response = content_filter.filter_response(response_data['answer'])
        
        return QueryResponse(
            answer=filtered_response['filtered_response'],
            confidence=response_data.get('confidence', 'unknown'),
            context_used=response_data.get('context_used', 0),
            sources=response_data.get('sources', []) if request.include_sources else None,
            response_time=end_time - start_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing query"
        )


# Configuration endpoints
@app.post("/system/configure", response_model=StandardResponse)
async def configure_pipeline(
    request: ConfigRequest,
    user: Dict[str, Any] = Depends(require_permission(Permission.CONFIGURE_SYSTEM))
):
    """Configure and initialize the RAG pipeline."""
    try:
        global rag_pipeline
        
        config = request.dict()
        config['data_dir'] = './data'
        
        # Validate configuration
        validation_result = input_validator.validate_configuration(config)
        if not validation_result['is_valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration: {validation_result['errors']}"
            )
        
        # Initialize pipeline
        rag_pipeline = RAGPipeline(**validation_result['sanitized_config'])
        
        return StandardResponse(
            success=True,
            message="Pipeline configured successfully",
            data={"config": validation_result['sanitized_config']}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error configuring pipeline"
        )


@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info(user: Dict[str, Any] = Depends(get_current_user)):
    """Get system information and statistics."""
    try:
        pipeline_info = {}
        health_checks = {
            "pipeline": rag_pipeline is not None,
            "input_validator": input_validator is not None,
            "content_filter": content_filter is not None,
            "access_controller": access_controller is not None
        }
        
        if rag_pipeline:
            pipeline_info = rag_pipeline.get_pipeline_info()
            health_checks["llm_loaded"] = pipeline_info.get('llm_info', {}).get('status') == 'loaded'
        
        performance_metrics = {
            "active_sessions": len(access_controller.sessions),
            "total_users": len(access_controller.users),
            "uptime": "N/A"  # Would need to track application start time
        }
        
        return SystemInfoResponse(
            pipeline_info=pipeline_info,
            system_health=health_checks,
            performance_metrics=performance_metrics
        )
    
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving system information"
        )


# Document management endpoints
@app.post("/documents/upload", response_model=StandardResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    user: Dict[str, Any] = Depends(require_permission(Permission.UPLOAD_DOCUMENTS))
):
    """Upload and process documents."""
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )
        
        temp_files = []
        processed_files = 0
        
        # Process uploaded files
        for file in files:
            # Validate file
            file_validation = input_validator.validate_file_path(file.filename)
            if not file_validation['is_valid']:
                logger.warning(f"Skipping invalid file {file.filename}: {file_validation['errors']}")
                continue
            
            # Save to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)
                processed_files += 1
        
        if not temp_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid files to process"
            )
        
        # Ingest documents
        result = rag_pipeline.ingest_documents(file_paths=temp_files)
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
        
        if result['status'] == 'success':
            return StandardResponse(
                success=True,
                message="Documents processed successfully",
                data={
                    "files_processed": result['documents_processed'],
                    "chunks_created": result['chunks_created'],
                    "statistics": result['statistics']
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {result}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing uploaded documents"
        )


@app.post("/documents/ingest", response_model=StandardResponse)
async def ingest_documents(
    request: DocumentIngestRequest,
    user: Dict[str, Any] = Depends(require_permission(Permission.UPLOAD_DOCUMENTS))
):
    """Ingest documents from file paths or directory."""
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )
        
        # Validate input
        if not request.file_paths and not request.directory_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either file_paths or directory_path must be provided"
            )
        
        # Ingest documents
        result = rag_pipeline.ingest_documents(
            file_paths=request.file_paths,
            directory_path=request.directory_path,
            file_extensions=request.file_extensions
        )
        
        if result['status'] == 'success':
            return StandardResponse(
                success=True,
                message="Documents ingested successfully",
                data=result
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document ingestion failed: {result}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document ingestion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error ingesting documents"
        )


@app.delete("/documents/clear", response_model=StandardResponse)
async def clear_documents(
    user: Dict[str, Any] = Depends(require_permission(Permission.DELETE_DOCUMENTS))
):
    """Clear all documents from the knowledge base."""
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not initialized"
            )
        
        rag_pipeline.reset_pipeline()
        
        return StandardResponse(
            success=True,
            message="All documents cleared successfully"
        )
    
    except Exception as e:
        logger.error(f"Document clearing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error clearing documents"
        )


# User management endpoints
@app.get("/users", response_model=StandardResponse)
async def list_users(
    user: Dict[str, Any] = Depends(require_permission(Permission.MANAGE_USERS))
):
    """List all users (admin only)."""
    try:
        users = access_controller.list_users()
        
        return StandardResponse(
            success=True,
            message="Users retrieved successfully",
            data={"users": users}
        )
    
    except Exception as e:
        logger.error(f"User listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving users"
        )


@app.get("/users/me", response_model=StandardResponse)
async def get_current_user_info(user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information."""
    try:
        user_info = access_controller.get_user_info(user['username'])
        
        return StandardResponse(
            success=True,
            message="User information retrieved successfully",
            data={"user": user_info}
        )
    
    except Exception as e:
        logger.error(f"User info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving user information"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
