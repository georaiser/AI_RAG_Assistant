# app.py
"""
Simplified Streamlit application for Technical Document Assistant.
Fixed vector store initialization issues with simpler approach.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, Any

from config import Config
from data_loader import load_data
from vector_store import VectorStoreManager
from backend import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="📚",
    layout="wide"
)


def initialize_app():
    """Initialize application components - simple approach."""
    if "app_initialized" in st.session_state:
        return st.session_state.vector_manager, st.session_state.rag_engine
    
    try:
        # Create vector manager
        vector_manager = VectorStoreManager()
        
        # Try to load existing vector store first
        if vector_manager.load_vector_store():
            logger.info("Loaded existing vector store")
            rag_engine = RAGEngine(vector_manager)
            
            # Store in session state
            st.session_state.vector_manager = vector_manager
            st.session_state.rag_engine = rag_engine
            st.session_state.vector_store_ready = True
            st.session_state.app_initialized = True
            
            # Get actual file info from data directory without reprocessing documents
            doc_count = vector_manager.get_document_count()
            try:
                # Get list of actual files from the documents directory
                from pathlib import Path
                documents_dir = Config.DOCUMENTS_DIR
                actual_files = []
                if documents_dir.exists():
                    for file_path in documents_dir.rglob("*"):
                        if (file_path.is_file() and 
                            file_path.suffix.lower() in Config.SUPPORTED_FORMATS and
                            not file_path.name.startswith('.')):
                            actual_files.append(file_path.name)
                
                st.session_state.file_info = {
                    "loaded_files": actual_files if actual_files else ["Existing documents (loaded from vector store)"],
                    "total_files": len(actual_files) if actual_files else 1,
                    "total_chunks": doc_count,
                    "file_stats": {"vector_store": {"chunks": doc_count}}
                }
            except Exception as e:
                logger.warning(f"Could not get file list: {e}")
                st.session_state.file_info = {
                    "loaded_files": ["Existing documents (loaded from vector store)"],
                    "total_files": 1,
                    "total_chunks": doc_count,
                    "file_stats": {"vector_store": {"chunks": doc_count}}
                }
            
            return vector_manager, rag_engine
        
        # If no existing store, create new one
        logger.info("Creating new vector store")
        documents, file_info = load_data()
        
        if not documents:
            st.error("No documents found in data directory")
            return None, None
        
        if vector_manager.initialize_vector_store(documents):
            rag_engine = RAGEngine(vector_manager)
            
            # Store in session state
            st.session_state.vector_manager = vector_manager
            st.session_state.rag_engine = rag_engine
            st.session_state.vector_store_ready = True
            st.session_state.file_info = file_info
            st.session_state.app_initialized = True
            
            logger.info("New vector store created successfully")
            return vector_manager, rag_engine
        else:
            st.error("Failed to create vector store")
            return None, None
            
    except Exception as e:
        logger.error(f"Error initializing app: {e}")
        st.error(f"Initialization error: {str(e)}")
        return None, None


class TechnicalDocumentApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        defaults = {
            "messages": [{"role": "assistant", "content": self._get_welcome_message()}],
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "last_query_info": {},
            "vector_store_ready": False,
            "file_info": {},
            "document_summary": ""
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_sidebar(self):
        """Render sidebar with status and controls."""
        with st.sidebar:
            # Status indicator
            if st.session_state.get("vector_store_ready", False):
                st.success("✅ Vector Store Ready")
            else:
                st.warning("⏳ Initializing...")
            
            st.divider()
            
            # File Information
            st.header("📁 Documents")
            file_info = st.session_state.get("file_info", {})
            if file_info:
                st.metric("Files", file_info.get("total_files", 0))
                st.metric("Chunks", file_info.get("total_chunks", 0))
                
                loaded_files = file_info.get("loaded_files", [])
                if loaded_files:
                    with st.expander("File List"):
                        for filename in loaded_files:
                            st.text(f"📄 {filename}")
                else:
                    st.info("📄 Vector store loaded with existing documents")
            else:
                st.info("📄 Loading document information...")
            
            st.divider()
            
            # Summary button
            if st.button("Generate Summary", disabled=not st.session_state.get("vector_store_ready", False)):
                self._generate_summary()
            
            # Clear conversation button
            if st.button("Clear Conversation", type="secondary"):
                st.session_state.messages = [{"role": "assistant", "content": self._get_welcome_message()}]
                st.rerun()
            
            if st.session_state.get("document_summary"):
                with st.expander("Summary"):
                    st.markdown(st.session_state.document_summary)
            
            st.divider()
            
            # Stats
            st.header("📊 Usage")
            st.text(f"Total Tokens: {st.session_state.total_tokens}")
            st.text(f"Total Cost: ${st.session_state.total_cost_usd:.6f}")
    
    def render_chat_interface(self):
        """Render main chat interface."""
        st.title(Config.APP_TITLE)
        
        # Get initialized components from session state
        vector_manager = st.session_state.get("vector_manager")
        rag_engine = st.session_state.get("rag_engine")
        
        if not vector_manager or not rag_engine:
            st.error("Failed to initialize. Please check your configuration and data directory.")
            return
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your documents..."):
            self._process_user_message(prompt)
    
    def _process_user_message(self, user_input: str, rag_engine: RAGEngine = None):
        """Process user message and generate response."""
        # Get RAG engine from session state if not provided
        if rag_engine is None:
            rag_engine = st.session_state.get("rag_engine")
            if not rag_engine:
                st.error("RAG engine not available")
                return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response_data = rag_engine.process_query(user_input, st.session_state.messages)
                st.markdown(response_data["answer"])
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response_data["answer"]})
        
        # Update stats
        self._update_statistics(response_data)
    
    def _generate_summary(self):
        """Generate document summary."""
        rag_engine = st.session_state.get("rag_engine")
        if not rag_engine:
            st.error("RAG engine not available")
            return
        
        with st.spinner("Generating summary..."):
            try:
                response_data = rag_engine.generate_summary()
                st.session_state.document_summary = response_data['answer']
                self._update_statistics(response_data)
                st.success("Summary generated!")
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
    
    def _update_statistics(self, response_data: Dict[str, Any]):
        """Update usage statistics."""
        tokens = response_data.get("total_tokens", 0)
        cost = response_data.get("total_cost_usd", 0.0)
        
        st.session_state.total_tokens += tokens
        st.session_state.total_cost_usd += cost
        st.session_state.last_query_info = {
            "total_tokens": tokens,
            "total_cost_usd": cost
        }
    
    def _get_welcome_message(self) -> str:
        """Get welcome message."""
        return f"""# Welcome to {Config.APP_TITLE}

I'm your technical documentation assistant.

## Capabilities:
- 🔍 Technical document analysis
- 💡 Code explanation with examples  
- 📊 Data and table interpretation
- 📋 Comprehensive summaries
- 🎯 Precise answers with citations

All responses include proper source citations.

What would you like to explore?"""
    
    def run(self):
        """Main application entry point."""
        if not Config.validate():
            st.error("Configuration invalid. Check your settings.")
            return
        
        if not Config.setup_directories():
            st.error("Failed to create directories.")
            return
        
        # Initialize app components first
        vector_manager, rag_engine = initialize_app()
        
        if not vector_manager or not rag_engine:
            st.error("Failed to initialize. Please check your configuration and data directory.")
            return
        
        self.render_sidebar()
        self.render_chat_interface()


def main():
    """Main function."""
    app = TechnicalDocumentApp()
    app.run()


if __name__ == "__main__":
    main()