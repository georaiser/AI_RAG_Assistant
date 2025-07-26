"""
Streamlit app for DocuPy Bot with multi-format document support.
"""

import streamlit as st
import logging
from typing import Dict, Any, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy library logs
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

# Import modules
from src.backend import handle_query
from src.data_loader import load_and_process_document, DocumentLoaderFactory
from src.vector_store import initialize_vector_store, load_vector_store
from src.config import config

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DocuPy Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_document_files() -> List[Path]:
    """Get list of supported document files."""
    # Try DOCUMENTS_DIR first, fallback to single file paths
    if hasattr(config, 'DOCUMENTS_DIR'):
        documents_dir = Path(config.DOCUMENT_PATH)
        if documents_dir.exists():
            supported_extensions = DocumentLoaderFactory.supported_extensions()
            document_files = []
            for ext in supported_extensions:
                document_files.extend(documents_dir.glob(f"*{ext}"))
            return sorted(document_files)
    
    # Fallback to single document
    for attr in ['DOCUMENT_PATH', 'PDF_PATH']:
        if hasattr(config, attr):
            file_path = Path(getattr(config, attr))
            if file_path.exists():
                return [file_path]
    
    return []

def initialize_app() -> bool:
    """Initialize the application with vector store."""
    try:
        # Check if vector store exists
        if not config.VECTOR_STORE_PATH.exists() or not any(config.VECTOR_STORE_PATH.iterdir()):
            logger.info("Creating new vector store...")
            
            with st.spinner("Initializing vector store..."):
                # Get document files
                document_files = get_document_files()
                if not document_files:
                    st.error("No supported document files found.")
                    return False
                
                st.info(f"Processing {len(document_files)} document(s)...")
                
                # Load all documents - FIX: Get Document objects, not just strings
                all_documents = []
                for file_path in document_files:
                    try:
                        docs = load_and_process_document(file_path)
                        if docs:
                            all_documents.extend(docs)  # docs are already Document objects
                            logger.info(f"Loaded {len(docs)} chunks from {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {str(e)}")
                        st.warning(f"Failed to load {file_path.name}")
                
                if not all_documents:
                    st.error("Failed to load any documents")
                    return False
                
                st.info(f"Creating embeddings for {len(all_documents)} chunks...")
                
                # Initialize vector store - FIX: Pass Document objects directly
                vector_store = initialize_vector_store(all_documents)
                if not vector_store:
                    st.error("Failed to create vector store")
                    return False
                
                st.success("Vector store created successfully!")
        else:
            logger.info("Loading existing vector store...")
            vector_store = load_vector_store()
            if not vector_store:
                st.error("Failed to load existing vector store")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        st.error(f"Initialization failed: {str(e)}")
        return False

def update_session_stats(result_data: Dict[str, Any]) -> None:
    """Update session statistics."""
    st.session_state.total_tokens += result_data.get("total_tokens", 0)
    st.session_state.total_cost_usd += result_data.get("total_cost_usd", 0.0)
    st.session_state.query_count += 1
    st.session_state.last_query_info = result_data

def render_sidebar() -> None:
    """Render sidebar with statistics and info."""
    with st.sidebar:
        st.title("Usage Statistics")
        
        # Last query information
        st.subheader("Last Query")
        if st.session_state.last_query_info:
            info = st.session_state.last_query_info
            st.metric("Tokens Used", info.get("total_tokens", 0))
            st.metric("Cost (USD)", f"${info.get('total_cost_usd', 0):.6f}")
            
            # Sources
            sources = info.get("sources", [])
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for source in sources:
                        st.text(f"• {source}")
        else:
            st.info("No queries processed yet")
        
        # Session totals
        st.subheader("Session Totals")
        st.metric("Total Queries", st.session_state.query_count)
        st.metric("Total Tokens", st.session_state.total_tokens)
        st.metric("Total Cost (USD)", f"${st.session_state.total_cost_usd:.6f}")
        
        if st.session_state.query_count > 0:
            avg_cost = st.session_state.total_cost_usd / st.session_state.query_count
            st.metric("Average Cost/Query", f"${avg_cost:.6f}")
        
        # Configuration
        st.subheader("Configuration")
        st.text(f"Model: {getattr(config, 'MODEL_NAME', 'Not set')}")
        st.text(f"Temperature: {getattr(config, 'TEMPERATURE', 'Not set')}")
        st.text(f"Retrieval K: {getattr(config, 'RETRIEVAL_K', 'Not set')}")
        st.text(f"Chunk Size: {getattr(config, 'CHUNK_SIZE', 'Not set')}")
        
        # Document info
        st.subheader("Document Library")
        document_files = get_document_files()
        if document_files:
            st.metric("Documents", len(document_files))
            supported_formats = DocumentLoaderFactory.supported_extensions()
            st.text(f"Supported: {', '.join(supported_formats)}")
            
            with st.expander("Document Details"):
                for file_path in document_files:
                    file_size = file_path.stat().st_size / 1024  # KB
                    st.text(f"{file_path.name} ({file_size:.1f} KB)")
        else:
            st.warning("No documents found")
        
        # Reset button
        if st.button("Reset Statistics"):
            st.session_state.total_tokens = 0
            st.session_state.total_cost_usd = 0.0
            st.session_state.query_count = 0
            st.session_state.last_query_info = {}
            st.rerun()

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        document_files = get_document_files()
        supported_formats = ", ".join(DocumentLoaderFactory.supported_extensions())
        
        welcome_message = f"""**Welcome to DocuPy Bot!**

I'm your document assistant with advanced RAG capabilities.

**Supported formats:** {supported_formats}
**Documents loaded:** {len(document_files) if document_files else 0}

**I can help you:**
- Find specific information in your documents
- Summarize content and key points
- Answer questions with context
- Extract relevant passages

Ask me anything about your documents!"""

        st.session_state.messages = [{
            "role": "assistant",
            "content": welcome_message
        }]
    
    # Initialize statistics
    defaults = {
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "query_count": 0,
        "last_query_info": {},
        "app_initialized": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("DocuPy Bot")
    st.markdown("*Multi-Format Document Assistant with RAG*")
    
    # Initialize app
    if not st.session_state.app_initialized:
        if initialize_app():
            st.session_state.app_initialized = True
            st.rerun()
        else:
            st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                try:
                    # Handle query
                    result_data = handle_query(user_input, st.session_state.messages)
                    response = result_data.get("answer", "I couldn't process your query.")
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Update statistics
                    update_session_stats(result_data)
                    
                    # Show sources if available
                    sources = result_data.get("sources", [])
                    if sources:
                        with st.expander(f"Sources Used ({len(sources)})"):
                            for source in sources:
                                st.text(f"• {source}")
                    
                    # Rerun to update sidebar
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Query processing error: {str(e)}")
                    error_msg = "Error processing your query. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()