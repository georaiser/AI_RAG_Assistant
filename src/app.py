"""
Streamlit app for Docu Bot.
"""

import streamlit as st
import logging
from typing import Dict, Any, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy logs
for lib in ["chromadb", "httpx", "openai", "langchain"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

from src.backend import handle_query
from src.data_loader import DocumentLoader, DocumentLoaderFactory
from src.vector_store import initialize_vector_store, load_vector_store
from src.config import config

loader = DocumentLoader()

# Page config
st.set_page_config(
    page_title="DocuPy Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_document_files() -> List[Path]:
    """Get list of supported document files - cached for performance."""
    document_path = Path(config.DOCUMENT_PATH)
    
    if document_path.is_file():
        return [document_path]
    elif document_path.is_dir():
        supported_exts = DocumentLoaderFactory.supported_extensions()
        files = []
        for ext in supported_exts:
            files.extend(document_path.glob(f"*{ext}"))
        return sorted(files)
    
    return []

def initialize_app() -> bool:
    """Initialize the application - simplified version."""
    try:
        # Check existing vector store
        if config.VECTOR_STORE_PATH.exists() and any(config.VECTOR_STORE_PATH.iterdir()):
            logger.info("Loading existing vector store...")
            vector_store = load_vector_store()
            return vector_store is not None
        
        # Create new vector store
        logger.info("Creating new vector store...")
        with st.spinner("Processing documents..."):
            document_files = get_document_files()
            
            if not document_files:
                st.error("No supported documents found.")
                return False
            
            # Load all documents
            all_documents = []
            for file_path in document_files:
                try:
                    docs = loader.load_and_process_document(Path(config.DOCUMENT_PATH))
                    all_documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} chunks from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    st.warning(f"Failed to load {file_path.name}")
            
            if not all_documents:
                st.error("Failed to load any documents")
                return False
            
            # Create vector store
            vector_store = initialize_vector_store(all_documents)
            if vector_store:
                st.success("Vector store created successfully!")
                return True
            else:
                st.error("Failed to create vector store")
                return False
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        st.error(f"Initialization failed: {e}")
        return False

def render_sidebar() -> None:
    """Simplified sidebar with essential info."""
    with st.sidebar:
        st.title("📊 Statistics")
        
        # Session stats
        st.metric("Queries", st.session_state.get('query_count', 0))
        st.metric("Total Tokens", st.session_state.get('total_tokens', 0))
        st.metric("Cost (USD)", f"${st.session_state.get('total_cost', 0):.4f}")
        
        # Last query info
        if last_info := st.session_state.get('last_query_info'):
            with st.expander("Last Query Details"):
                st.write(f"**Tokens:** {last_info.get('total_tokens', 0)}")
                st.write(f"**Cost:** ${last_info.get('total_cost_usd', 0):.6f}")
                
                if sources := last_info.get('sources'):
                    st.write(f"**Sources:** {len(sources)}")
                    for source in sources[:3]:  # Show max 3
                        st.caption(source)
        
        st.divider()
        
        # Document info
        st.subheader("📚 Documents")
        documents = get_document_files()
        st.metric("Files", len(documents))
        
        if documents:
            with st.expander("Document List"):
                for doc in documents:
                    size_kb = doc.stat().st_size / 1024
                    st.caption(f"{doc.name} ({size_kb:.1f} KB)")
        
        # Reset button
        if st.button("🔄 Reset Stats", type="secondary"):
            for key in ['query_count', 'total_tokens', 'total_cost', 'last_query_info']:
                st.session_state[key] = 0 if key != 'last_query_info' else {}
            st.rerun()

def initialize_session_state() -> None:
    """Initialize session state with defaults."""
    if "messages" not in st.session_state:
        documents = get_document_files()
        supported = ", ".join(DocumentLoaderFactory.supported_extensions())
        
        welcome = f"""**Welcome to DocuPy Bot!** 🤖

I'm your document assistant powered by advanced RAG.

**📄 Documents loaded:** {len(documents)}
**📝 Supported formats:** {supported}

**What I can help with:**
- Answer questions about your documents
- Find specific information quickly  
- Summarize content and extract key points
- Provide contextual explanations

Ask me anything about your documents!"""

        st.session_state.messages = [{"role": "assistant", "content": welcome}]
    
    # Initialize counters
    for key, default in [
        ("query_count", 0), ("total_tokens", 0), 
        ("total_cost", 0.0), ("app_initialized", False)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

def main():
    """Main application - streamlined."""
    initialize_session_state()
    
    # Header
    st.title("🤖 DocuPy Bot")
    st.caption("*Intelligent Document Assistant with RAG*")
    
    # Initialize app once
    if not st.session_state.app_initialized:
        if initialize_app():
            st.session_state.app_initialized = True
            st.rerun()
        else:
            st.stop()
    
    # Sidebar
    render_sidebar()
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                try:
                    # Handle query
                    result = handle_query(user_input, st.session_state.messages)
                    response = result.get("answer", "I couldn't process your query.")
                    
                    # Display response
                    st.markdown(response)
                    
                    # Update session
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.query_count += 1
                    st.session_state.total_tokens += result.get("total_tokens", 0)
                    st.session_state.total_cost += result.get("total_cost_usd", 0.0)
                    st.session_state.last_query_info = result
                    
                    # Show sources
                    if sources := result.get("sources"):
                        with st.expander(f"📖 Sources ({len(sources)})"):
                            for source in sources:
                                st.caption(f"• {source}")
                    
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Query error: {e}")
                    error_msg = "❌ Error processing query. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()