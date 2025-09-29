@echo off
REM Windows PowerShell deployment script for RAG Chatbot

set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set NC=[0m

echo %GREEN%[INFO]%NC% RAG Chatbot Docker Management (Windows)

:check_docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERROR]%NC% Docker is not running. Please start Docker Desktop and try again.
    exit /b 1
)
echo %GREEN%[INFO]%NC% Docker is running

:main
if "%1"=="start-prod" goto start_prod
if "%1"=="start-dev" goto start_dev
if "%1"=="stop" goto stop_services
if "%1"=="status" goto show_status
if "%1"=="logs" goto show_logs
if "%1"=="cleanup" goto cleanup
if "%1"=="help" goto show_help
if "%1"=="" goto show_help
goto show_help

:start_prod
echo %GREEN%[INFO]%NC% Starting RAG Chatbot in production mode...
call :create_directories
docker-compose down
docker-compose build --no-cache
docker-compose up -d
echo %GREEN%[INFO]%NC% Services starting up...
timeout /t 10 /nobreak >nul
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%[INFO]%NC% ✅ API service is healthy
) else (
    echo %RED%[ERROR]%NC% ❌ API service health check failed
)
echo %GREEN%[INFO]%NC% RAG Chatbot started successfully!
echo %GREEN%[INFO]%NC% API: http://localhost:8000
echo %GREEN%[INFO]%NC% Web Interface: http://localhost:8501
echo %GREEN%[INFO]%NC% API Docs: http://localhost:8000/docs
goto end

:start_dev
echo %GREEN%[INFO]%NC% Starting RAG Chatbot in development mode...
call :create_directories
docker-compose -f docker-compose.dev.yml down
docker-compose -f docker-compose.dev.yml build --no-cache
docker-compose -f docker-compose.dev.yml up -d
echo %GREEN%[INFO]%NC% Development environment started!
echo %GREEN%[INFO]%NC% API (with hot reload): http://localhost:8000
echo %GREEN%[INFO]%NC% Web Interface (with hot reload): http://localhost:8501
goto end

:stop_services
echo %GREEN%[INFO]%NC% Stopping RAG Chatbot services...
docker-compose down
docker-compose -f docker-compose.dev.yml down 2>nul
echo %GREEN%[INFO]%NC% Services stopped
goto end

:show_status
echo %GREEN%[INFO]%NC% RAG Chatbot Service Status:
docker-compose ps
echo.
echo %GREEN%[INFO]%NC% Service Health:
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo API: %GREEN%✅ Healthy%NC%
) else (
    echo API: %RED%❌ Unhealthy%NC%
)
curl -f http://localhost:8501 >nul 2>&1
if %errorlevel% equ 0 (
    echo Web: %GREEN%✅ Accessible%NC%
) else (
    echo Web: %RED%❌ Inaccessible%NC%
)
goto end

:show_logs
if "%2"=="" (
    docker-compose logs -f
) else (
    docker-compose logs -f %2
)
goto end

:cleanup
echo %YELLOW%[WARN]%NC% This will remove all RAG Chatbot containers, images, and volumes
set /p confirm=Are you sure? (y/N): 
if /i "%confirm%"=="y" (
    echo %GREEN%[INFO]%NC% Cleaning up Docker resources...
    docker-compose down -v --rmi all --remove-orphans
    docker system prune -f
    echo %GREEN%[INFO]%NC% Cleanup completed
) else (
    echo %GREEN%[INFO]%NC% Cleanup cancelled
)
goto end

:create_directories
echo %GREEN%[INFO]%NC% Creating necessary directories...
if not exist "data\chroma_db" mkdir data\chroma_db
if not exist "data\faiss_index" mkdir data\faiss_index
if not exist "logs" mkdir logs
if not exist "config" mkdir config
echo %GREEN%[INFO]%NC% Directories created
exit /b 0

:show_help
echo RAG Chatbot Docker Management Script (Windows)
echo.
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   start-prod         Start in production mode
echo   start-dev          Start in development mode
echo   stop              Stop all services
echo   status            Show service status
echo   logs [service]    View logs (optionally for specific service)
echo   cleanup           Remove all Docker resources
echo   help             Show this help message
echo.
echo Examples:
echo   %0 start-prod     # Start production environment
echo   %0 logs rag-api   # View API logs
goto end

:end
