"""
Main entry point for the RAG chatbot application.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

if __name__ == "__main__":
    print("🤖 RAG Chatbot - Retrieval-Augmented Generation System")
    print("=" * 60)
    print()
    print("Available interfaces:")
    print("1. Web Interface (Streamlit): streamlit run src/web_interface/streamlit_app.py")
    print("2. REST API (FastAPI): uvicorn src.web_interface.fastapi_app:app --reload")
    print("3. Docker Deployment: ./docker/deploy.sh start-prod")
    print()
    print("For detailed instructions, see README.md or INSTALL.md")
    print()
    
    # Quick system check
    try:
        from src.rag_pipeline.rag_pipeline import RAGPipeline
        from src.governance.input_validation import InputValidator
        print("✅ Core modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    
    print()
    print("Repository: https://github.com/your-username/rag-chatbot")
    print("Documentation: http://localhost:8000/docs (when API is running)")
