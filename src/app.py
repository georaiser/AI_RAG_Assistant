# app.py
"""
Main Streamlit application for Technical Document Assistant.
Provides web interface for document querying and summarization.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, Any

from config import Config
from data_loader import load_data
from vector_store import VectorStoreManager
from backend import RAGEngine, EmbeddingManager

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


class TechnicalDocumentApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_vector_store()
    
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
    
    def setup_vector_store(self):
        """Setup vector store if not exists."""
        if not st.session_state.vector_store_ready:
            vector_manager = VectorStoreManager()
            
            # Check if vector store exists
            if not Config.VECTOR_STORE_PATH.exists() or not any(Config.VECTOR_STORE_PATH.iterdir()):
                logger.info("Initializing vector store for the first time")
                st.info("Initializing vector store for the first time. This may take a few minutes...")
                
                # Load and process documents
                documents, file_info = load_data()
                if documents:
                    if vector_manager.initialize_vector_store(documents):
                        st.session_state.vector_store_ready = True
                        st.session_state.file_info = file_info
                        logger.info("Vector store initialized successfully")
                        st.success("Vector store initialized successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to initialize vector store")
                        logger.error("Failed to initialize vector store")
                else:
                    st.error("No documents found to process")
                    logger.error("No documents found to process")
            else:
                # Load existing vector store
                if vector_manager.load_vector_store():
                    st.session_state.vector_store_ready = True
                    # Try to load file info from previous session or regenerate
                    if not st.session_state.file_info:
                        try:
                            _, file_info = load_data()
                            st.session_state.file_info = file_info
                        except Exception as e:
                            logger.warning(f"Could not load file info: {e}")
                    logger.info("Vector store loaded successfully")
    
    def render_sidebar(self):
        """Render sidebar with file information, statistics and summary."""
        with st.sidebar:
            # File Information
            st.header("📁 Loaded Documents")
            if st.session_state.file_info:
                info = st.session_state.file_info
                st.metric("Total Files", info.get("total_files", 0))
                st.metric("Total Chunks", info.get("total_chunks", 0))
                
                if info.get("loaded_files"):
                    with st.expander("File Details", expanded=False):
                        for filename in info["loaded_files"]:
                            stats = info["file_stats"].get(filename, {})
                            st.text(f"📄 {filename}")
                            st.text(f"  Chunks: {stats.get('chunks', 0)}")
                            st.text(f"  Tables: {stats.get('tables', 0)}")
                            st.text(f"  Size: {stats.get('size_mb', 0):.1f} MB")
                            st.text("")
            else:
                st.text("No documents loaded yet")
            
            st.divider()
            
            # Document Summary Section
            st.header("📋 Document Summary")
            if st.button("Generate Summary", type="primary", use_container_width=True):
                self._generate_summary()
            
            if st.session_state.document_summary:
                with st.expander("View Summary", expanded=False):
                    st.markdown(st.session_state.document_summary)
            
            st.divider()
            
            # Usage Statistics
            st.header("📊 Usage Statistics")
            
            # Last query info
            st.subheader("Last Query")
            if st.session_state.last_query_info:
                info = st.session_state.last_query_info
                st.text(f"Tokens: {info.get('total_tokens', 0)}")
                st.text(f"Cost: ${info.get('total_cost_usd', 0):.6f}")
            else:
                st.text("No queries processed yet")
            
            # Session totals
            st.subheader("Session Total")
            st.text(f"Tokens: {st.session_state.total_tokens}")
            st.text(f"Cost: ${st.session_state.total_cost_usd:.6f}")
            
            st.divider()
            
            # Configuration options
            self._render_config_options()
    
    def _render_config_options(self):
        """Render configuration options in sidebar."""
        st.header("⚙️ Configuration")
        
        # Embedding model selection
        embedding_models = EmbeddingManager.get_available_models()
        
        model_type = st.selectbox(
            "Embedding Type",
            options=["OpenAI", "HuggingFace"],
            index=0 if Config.EMBEDDING_TYPE == "openai" else 1
        )
        
        available_models = embedding_models[model_type]
        current_model = (Config.OPENAI_EMBEDDING_MODEL if model_type == "OpenAI" 
                        else Config.HF_EMBEDDING_MODEL)
        
        model_name = st.selectbox(
            "Embedding Model",
            options=available_models,
            index=available_models.index(current_model) if current_model in available_models else 0
        )
        
        if st.button("Update Model", use_container_width=True):
            if EmbeddingManager.update_embedding_model(model_type.lower(), model_name):
                st.success("Model updated! Restart to apply changes.")
        
        # Search strategy
        search_type = st.selectbox(
            "Search Strategy",
            options=Config.AVAILABLE_SEARCH_TYPES,
            index=Config.AVAILABLE_SEARCH_TYPES.index(Config.SEARCH_TYPE)
        )
        
        if search_type != Config.SEARCH_TYPE:
            Config.SEARCH_TYPE = search_type
            st.success(f"Search strategy: {search_type}")
    
    def render_chat_interface(self):
        """Render main chat interface."""
        st.title(Config.APP_TITLE)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a technical question about your documents..."):
            if not st.session_state.vector_store_ready:
                st.error("Vector store is not ready. Please wait for initialization.")
                return
            
            self._process_user_message(prompt)
    
    def _process_user_message(self, user_input: str):
        """Process user message and generate response."""
        logger.info(f"Processing user query: {user_input[:50]}...")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                response_data = self._get_assistant_response(user_input)
                st.markdown(response_data["answer"])
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response_data["answer"]})
        
        # Update statistics
        self._update_statistics(response_data)
        
        # Rerun to update sidebar
        st.rerun()
    
    def _get_assistant_response(self, user_input: str) -> Dict[str, Any]:
        """Get response from RAG engine."""
        if not st.session_state.vector_store_ready:
            return {
                "answer": "Vector store is not ready. Please wait for initialization to complete.",
                "total_tokens": 0,
                "total_cost_usd": 0.0
            }
        
        try:
            engine = RAGEngine()
            response = engine.process_query(user_input, st.session_state.messages)
            logger.info(f"Query processed successfully. Tokens: {response.get('total_tokens', 0)}")
            return response
        except Exception as e:
            logger.error(f"Error in assistant response: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "total_tokens": 0,
                "total_cost_usd": 0.0
            }
    
    def _generate_summary(self):
        """Generate document summary."""
        if not st.session_state.vector_store_ready:
            st.error("Vector store is not ready")
            return
        
        logger.info("Generating document summary")
        with st.spinner("Generating comprehensive document summary..."):
            try:
                engine = RAGEngine()
                response_data = engine.generate_summary()
                
                # Store summary separately from chat
                st.session_state.document_summary = response_data['answer']
                
                # Update statistics
                self._update_statistics(response_data)
                
                st.success("Document summary generated!")
                logger.info("Document summary generated successfully")
                st.rerun()
                
            except Exception as e:
                error_msg = f"Failed to generate summary: {e}"
                st.error(error_msg)
                logger.error(error_msg)
    
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
        """Get welcome message for the application."""
        doc_mode = "multiple documents" if Config.PROCESS_MULTIPLE_DOCUMENTS else "single document"
        
        return f"""# Welcome to {Config.APP_TITLE}

I'm your technical documentation assistant, specialized in analyzing and explaining technical documents with precision.

## My Capabilities:
- 🔍 **Technical Analysis**: Deep analysis of technical documentation
- 💡 **Code Explanation**: Detailed code examples and implementations  
- 📊 **Data Interpretation**: Table and figure analysis
- 📋 **Documentation**: Comprehensive summaries with proper citations
- 🎯 **Precise Answers**: All responses include proper source citations

## Current Configuration:
- **Processing**: {doc_mode}
- **Embedding**: {Config.OPENAI_EMBEDDING_MODEL if Config.EMBEDDING_TYPE == 'openai' else Config.HF_EMBEDDING_MODEL}
- **Search**: {Config.SEARCH_TYPE}

Every answer includes proper citations like [Source: filename] or [Source: filename, Page: X]. Use the sidebar to view loaded documents and generate comprehensive summaries.

What technical topic would you like to explore?"""

    def run(self):
        """Main application entry point."""
        logger.info("Starting Technical Document Assistant")
        
        # Validate configuration
        if not Config.validate():
            st.error("Configuration validation failed. Please check your settings.")
            logger.error("Configuration validation failed")
            return
        
        # Setup directories
        if not Config.setup_directories():
            st.error("Failed to setup required directories.")
            logger.error("Failed to setup directories")
            return
        
        # Render interface
        self.render_sidebar()
        self.render_chat_interface()


def main():
    """Main function to run the Streamlit app."""
    app = TechnicalDocumentApp()
    app.run()


if __name__ == "__main__":
    main()