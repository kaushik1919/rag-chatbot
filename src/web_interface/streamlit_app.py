"""
Streamlit web interface for the RAG chatbot.
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import time

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


class StreamlitRAGApp:
    """Streamlit application for RAG chatbot."""
    
    def __init__(self):
        self.init_session_state()
        self.setup_components()
    
    def init_session_state(self):
        """Initialize Streamlit session state."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.chat_history = []
            st.session_state.rag_pipeline = None
            st.session_state.current_user = None
            st.session_state.session_token = None
            st.session_state.uploaded_files = []
    
    def setup_components(self):
        """Setup RAG components."""
        try:
            # Initialize governance components
            self.input_validator = InputValidator()
            self.content_filter = ContentFilter(FilterLevel.MODERATE)
            self.access_controller = AccessController()
            self.prompt_engineer = PromptEngineer()
            
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            logger.error(f"Component initialization error: {e}")
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
            text-align: center;
            color: #1e88e5;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #1e88e5;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left-color: #1976d2;
        }
        .assistant-message {
            background-color: #f5f5f5;
            border-left-color: #4caf50;
        }
        .error-message {
            background-color: #ffebee;
            border-left-color: #f44336;
            color: #c62828;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Check authentication
        if not self.check_authentication():
            self.show_login_page()
            return
        
        # Main application
        st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.show_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.show_chat_interface()
        
        with col2:
            self.show_system_info()
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated."""
        if st.session_state.session_token:
            # Validate existing session
            session_info = self.access_controller.validate_session(st.session_state.session_token)
            if session_info.get('valid', False):
                st.session_state.current_user = session_info
                return True
            else:
                # Session expired
                st.session_state.session_token = None
                st.session_state.current_user = None
        
        return False
    
    def show_login_page(self):
        """Show login interface."""
        st.markdown('<h1 class="main-header">🔐 Login</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Login")
                
                if submit_button:
                    if username and password:
                        # Validate credentials
                        auth_result = self.access_controller.authenticate_user(username, password)
                        
                        if auth_result['success']:
                            st.session_state.session_token = auth_result['session_token']
                            st.session_state.current_user = auth_result['user_info']
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error(f"Login failed: {auth_result['error']}")
                    else:
                        st.error("Please enter both username and password")
            
            # Guest access option
            if st.button("Continue as Guest"):
                # Create temporary guest session
                guest_result = self.access_controller.authenticate_user("guest", "guest_access")
                if not guest_result['success']:
                    # Create guest user if doesn't exist
                    self.access_controller.create_user("guest", "guest_access", UserRole.GUEST)
                    guest_result = self.access_controller.authenticate_user("guest", "guest_access")
                
                if guest_result['success']:
                    st.session_state.session_token = guest_result['session_token']
                    st.session_state.current_user = guest_result['user_info']
                    st.rerun()
    
    def show_sidebar(self):
        """Show sidebar with configuration and controls."""
        st.sidebar.header("Configuration")
        
        # User info
        if st.session_state.current_user:
            user_info = st.session_state.current_user
            st.sidebar.write(f"**User:** {user_info['username']}")
            st.sidebar.write(f"**Role:** {user_info['role']}")
        
        # Logout button
        if st.sidebar.button("Logout"):
            if st.session_state.session_token:
                self.access_controller.logout_user(st.session_state.session_token)
            st.session_state.session_token = None
            st.session_state.current_user = None
            st.rerun()
        
        st.sidebar.divider()
        
        # RAG Pipeline Configuration
        st.sidebar.subheader("Model Settings")
        
        # Check permissions for configuration
        can_configure = self.access_controller.check_permission(
            st.session_state.session_token, 
            Permission.CONFIGURE_SYSTEM
        )
        
        # LLM Model Selection
        llm_options = ['mistral-7b', 'llama2-7b', 'falcon-7b']
        selected_llm = st.sidebar.selectbox(
            "LLM Model", 
            llm_options, 
            disabled=not can_configure,
            help="Select the language model for generation"
        )
        
        # Embedding Model Selection
        embedding_options = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'multi-qa-MiniLM-L6-cos-v1']
        selected_embedding = st.sidebar.selectbox(
            "Embedding Model", 
            embedding_options,
            disabled=not can_configure,
            help="Select the embedding model for document retrieval"
        )
        
        # Vector Store Selection
        vector_store_options = ['chroma', 'faiss']
        selected_vector_store = st.sidebar.selectbox(
            "Vector Store", 
            vector_store_options,
            disabled=not can_configure,
            help="Select the vector database backend"
        )
        
        # Advanced Settings
        with st.sidebar.expander("Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, disabled=not can_configure)
            max_length = st.slider("Max Response Length", 100, 4000, 1000, 100, disabled=not can_configure)
            top_k = st.slider("Retrieved Documents", 1, 10, 5, disabled=not can_configure)
        
        # Initialize/Update Pipeline Button
        if st.sidebar.button("Initialize/Update Pipeline", disabled=not can_configure):
            self.initialize_pipeline(selected_llm, selected_embedding, selected_vector_store)
        
        st.sidebar.divider()
        
        # Document Management
        self.show_document_management()
        
        st.sidebar.divider()
        
        # Safety Settings
        with st.sidebar.expander("Safety Settings"):
            filter_levels = ['strict', 'moderate', 'relaxed']
            selected_filter = st.selectbox("Content Filter Level", filter_levels, index=1)
            
            if st.button("Update Filter Level"):
                self.content_filter.update_filter_level(FilterLevel(selected_filter))
                st.success(f"Filter level updated to {selected_filter}")
    
    def show_document_management(self):
        """Show document upload and management interface."""
        st.sidebar.subheader("Documents")
        
        # Check upload permission
        can_upload = self.access_controller.check_permission(
            st.session_state.session_token, 
            Permission.UPLOAD_DOCUMENTS
        )
        
        # File upload
        uploaded_files = st.sidebar.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx'],
            disabled=not can_upload,
            help="Upload documents to add to the knowledge base"
        )
        
        if uploaded_files and can_upload:
            if st.sidebar.button("Process Uploads"):
                self.process_uploaded_files(uploaded_files)
        
        # Show uploaded files count
        if st.session_state.rag_pipeline:
            pipeline_info = st.session_state.rag_pipeline.get_pipeline_info()
            doc_count = pipeline_info.get('document_count', 0)
            st.sidebar.metric("Documents in KB", doc_count)
        
        # Clear documents button (admin only)
        can_delete = self.access_controller.check_permission(
            st.session_state.session_token, 
            Permission.DELETE_DOCUMENTS
        )
        
        if can_delete:
            if st.sidebar.button("Clear All Documents", type="secondary"):
                if st.session_state.rag_pipeline:
                    st.session_state.rag_pipeline.reset_pipeline()
                    st.sidebar.success("All documents cleared")
    
    def show_chat_interface(self):
        """Show the main chat interface."""
        st.subheader("💬 Chat with your documents")
        
        # Check query permission
        can_query = self.access_controller.check_permission(
            st.session_state.session_token, 
            Permission.QUERY_SYSTEM
        )
        
        if not can_query:
            st.error("You don't have permission to query the system.")
            return
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(
                        f'<div class="chat-message user-message"><strong>You:</strong><br>{message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br>{message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show sources if available
                    if 'sources' in message:
                        with st.expander(f"Sources for response {i+1}"):
                            for j, source in enumerate(message['sources'][:3]):  # Show top 3 sources
                                st.write(f"**Source {j+1}** (Score: {source['score']:.3f})")
                                st.write(f"File: {source['metadata'].get('filename', 'Unknown')}")
                                st.write(f"Content: {source['metadata'].get('content', '')[:200]}...")
        
        # Chat input
        user_input = st.chat_input(
            "Ask a question about your documents...",
            disabled=not st.session_state.rag_pipeline
        )
        
        if user_input:
            self.process_user_message(user_input)
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            self.prompt_engineer.clear_conversation_history()
            st.rerun()
    
    def show_system_info(self):
        """Show system information and statistics."""
        st.subheader("📊 System Information")
        
        if st.session_state.rag_pipeline:
            pipeline_info = st.session_state.rag_pipeline.get_pipeline_info()
            
            # Model information
            st.write("**Current Configuration:**")
            st.write(f"- LLM Model: {pipeline_info.get('llm_model', 'Not loaded')}")
            st.write(f"- Embedding Model: {pipeline_info.get('embedding_model', 'Not loaded')}")
            st.write(f"- Vector Store: {pipeline_info.get('vector_store_type', 'Not configured')}")
            
            # Statistics
            st.write("**Statistics:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", pipeline_info.get('document_count', 0))
            with col2:
                vector_stats = pipeline_info.get('vector_store_stats', {})
                st.metric("Total Chunks", vector_stats.get('total_documents', 0))
        
        else:
            st.info("Pipeline not initialized. Please configure and initialize the system.")
        
        # System health
        st.subheader("🔧 System Health")
        
        health_checks = self.run_health_checks()
        for check_name, status in health_checks.items():
            if status:
                st.success(f"✅ {check_name}")
            else:
                st.error(f"❌ {check_name}")
        
        # Performance metrics
        if st.session_state.chat_history:
            st.subheader("📈 Performance")
            avg_response_time = self.calculate_avg_response_time()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    def initialize_pipeline(self, llm_model: str, embedding_model: str, vector_store: str):
        """Initialize or update the RAG pipeline."""
        try:
            with st.spinner("Initializing RAG pipeline..."):
                config = {
                    'llm_model': llm_model,
                    'embedding_model': embedding_model,
                    'vector_store_type': vector_store,
                    'data_dir': './data'
                }
                
                # Validate configuration
                validation_result = self.input_validator.validate_configuration(config)
                if not validation_result['is_valid']:
                    st.error(f"Invalid configuration: {validation_result['errors']}")
                    return
                
                # Initialize pipeline
                st.session_state.rag_pipeline = RAGPipeline(**validation_result['sanitized_config'])
                st.success("Pipeline initialized successfully!")
                
        except Exception as e:
            st.error(f"Error initializing pipeline: {e}")
            logger.error(f"Pipeline initialization error: {e}")
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and add to knowledge base."""
        if not st.session_state.rag_pipeline:
            st.error("Please initialize the pipeline first.")
            return
        
        try:
            with st.spinner("Processing uploaded files..."):
                temp_files = []
                
                # Save uploaded files to temporary directory
                for uploaded_file in uploaded_files:
                    # Validate file
                    file_validation = self.input_validator.validate_file_path(uploaded_file.name)
                    if not file_validation['is_valid']:
                        st.error(f"File {uploaded_file.name}: {file_validation['errors']}")
                        continue
                    
                    # Save to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_files.append(tmp_file.name)
                
                if temp_files:
                    # Ingest documents
                    result = st.session_state.rag_pipeline.ingest_documents(file_paths=temp_files)
                    
                    if result['status'] == 'success':
                        st.success(f"Successfully processed {result['documents_processed']} documents with {result['chunks_created']} chunks.")
                    else:
                        st.error(f"Error processing documents: {result}")
                    
                    # Clean up temporary files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except Exception:
                            pass
                
        except Exception as e:
            st.error(f"Error processing files: {e}")
            logger.error(f"File processing error: {e}")
    
    def process_user_message(self, user_input: str):
        """Process user message and generate response."""
        if not st.session_state.rag_pipeline:
            st.error("Please initialize the pipeline first.")
            return
        
        # Validate input
        validation_result = self.input_validator.validate_query(user_input)
        if not validation_result['is_valid']:
            st.error(f"Invalid input: {validation_result['errors']}")
            return
        
        # Safety check
        safety_result = self.content_filter.check_query_safety(user_input)
        if not safety_result['should_process']:
            st.error(f"Query blocked by content filter: {safety_result['warnings']}")
            return
        
        try:
            # Add user message to chat
            st.session_state.chat_history.append({
                'role': 'user',
                'content': validation_result['sanitized_query']
            })
            
            # Generate response
            start_time = time.time()
            
            with st.spinner("Generating response..."):
                response_data = st.session_state.rag_pipeline.query(
                    question=validation_result['sanitized_query'],
                    top_k=5,
                    include_sources=True
                )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Filter response
            filtered_response = self.content_filter.filter_response(response_data['answer'])
            
            # Add assistant message to chat
            assistant_message = {
                'role': 'assistant',
                'content': filtered_response['filtered_response'],
                'sources': response_data.get('sources', []),
                'confidence': response_data.get('confidence', 'unknown'),
                'response_time': response_time
            }
            
            if filtered_response['was_filtered']:
                assistant_message['content'] += "\n\n*Note: Some content was filtered for safety.*"
            
            st.session_state.chat_history.append(assistant_message)
            
            # Update conversation history for prompt engineering
            self.prompt_engineer.add_to_conversation(
                validation_result['sanitized_query'],
                filtered_response['filtered_response']
            )
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            logger.error(f"Response generation error: {e}")
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run system health checks."""
        checks = {}
        
        # Check if pipeline is loaded
        checks['Pipeline Loaded'] = st.session_state.rag_pipeline is not None
        
        # Check if model is loaded
        if st.session_state.rag_pipeline:
            try:
                llm_info = st.session_state.rag_pipeline.llm.get_model_info()
                checks['LLM Model Loaded'] = llm_info.get('status') == 'loaded'
            except:
                checks['LLM Model Loaded'] = False
        else:
            checks['LLM Model Loaded'] = False
        
        # Check governance components
        checks['Input Validator'] = self.input_validator is not None
        checks['Content Filter'] = self.content_filter is not None
        checks['Access Controller'] = self.access_controller is not None
        
        return checks
    
    def calculate_avg_response_time(self) -> float:
        """Calculate average response time from chat history."""
        response_times = [
            msg.get('response_time', 0) 
            for msg in st.session_state.chat_history 
            if msg['role'] == 'assistant' and 'response_time' in msg
        ]
        
        return sum(response_times) / len(response_times) if response_times else 0.0


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitRAGApp()
    app.run()


if __name__ == "__main__":
    main()
