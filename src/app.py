# app.py
"""
Streamlit RAG application with fixed role handling and improved citations.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, Any

from config import Config
from data_loader import load_data
from vector_store import VectorStoreManager
#from backend import RAGEngine # Uncomment this line to use langchain's RAGEngine
from langgraph_rag import LangGraphRAGEngine as RAGEngine # to use langgraph's RAGEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple role constants
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="📚",
    layout="wide"
)


def initialize_app():
    """Initialize application components."""
    if "app_initialized" in st.session_state:
        return st.session_state.vector_manager, st.session_state.rag_engine
    
    try:
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
            
            # Get actual file info from vector store
            doc_count = vector_manager.get_document_count()
            file_list = vector_manager.get_loaded_files()
            st.session_state.file_info = {
                "loaded_files": file_list,
                "total_files": len(file_list),
                "total_chunks": doc_count
            }
            
            return vector_manager, rag_engine
        
        # Create new vector store
        logger.info("Creating new vector store")
        documents, file_info = load_data()
        
        if not documents:
            st.error("No documents found in data directory")
            return None, None
        
        if vector_manager.initialize_vector_store(documents):
            rag_engine = RAGEngine(vector_manager)
            
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
    """Simplified application class."""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state."""
        defaults = {
            "messages": [],
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "last_query_info": {},
            "vector_store_ready": False,
            "file_info": {},
            "document_summary": "",
            "show_sources": Config.DEBUG_MODE
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Add welcome message only if no messages exist
        if not st.session_state.messages:
            st.session_state.messages = [
                {"role": ASSISTANT_ROLE, "content": self._get_welcome_message()}
            ]
    
    def render_sidebar(self):
        """Render simplified sidebar."""
        with st.sidebar:
            # Status indicator
            if st.session_state.get("vector_store_ready", False):
                st.success("Vector Store Ready")
            else:
                st.warning("Initializing...")
            
            st.divider()
            
            # File information with list
            st.header("Documents")
            file_info = st.session_state.get("file_info", {})
            if file_info:
                st.metric("Files", file_info.get("total_files", 0))
                st.metric("Chunks", file_info.get("total_chunks", 0))
                
                # Show file list
                loaded_files = file_info.get("loaded_files", [])
                if loaded_files:
                    with st.expander("File List", expanded=True):
                        for filename in loaded_files:
                            st.text(f"• {filename}")
            
            st.divider()
            
            # Options
            st.header("Options")
            
            # Show Sources toggle - now working
            st.session_state.show_sources = st.checkbox(
                "Show Sources", 
                value=st.session_state.get("show_sources", Config.DEBUG_MODE),
                help="Show retrieved document chunks used to generate the answer"
            )
            
            if st.button("Generate Summary", disabled=not st.session_state.get("vector_store_ready", False)):
                self._generate_summary()
            
            if st.button("Clear Conversation", type="secondary"):
                st.session_state.messages = [
                    {"role": ASSISTANT_ROLE, "content": self._get_welcome_message()}
                ]
                st.rerun()
            
            # Show summary if available
            if st.session_state.get("document_summary"):
                with st.expander("Document Summary"):
                    st.markdown(st.session_state.document_summary)
            
            st.divider()
            
            # Usage stats
            st.header("Usage Stats")
            st.text(f"Total Tokens: {st.session_state.total_tokens}")
            st.text(f"Total Cost: ${st.session_state.total_cost_usd:.6f}")
            
            # Last query info
            if st.session_state.get("last_query_info"):
                info = st.session_state.last_query_info
                if Config.VERBOSE_MODE:
                    with st.expander("Last Query Details"):
                        st.text(f"Tokens: {info.get('total_tokens', 0)}")
                        st.text(f"Cost: ${info.get('total_cost_usd', 0):.6f}")
                        st.text(f"Sources: {info.get('source_count', 0)}")
    
    def render_chat_interface(self):
        """Render simplified chat interface."""
        st.title(Config.APP_TITLE)
        
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
    
    def _process_user_message(self, user_input: str):
        """Process user message with improved error handling."""
        rag_engine = st.session_state.get("rag_engine")
        if not rag_engine:
            st.error("RAG engine not available")
            return
        
        # Add user message
        st.session_state.messages.append({"role": USER_ROLE, "content": user_input})
        
        with st.chat_message(USER_ROLE):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message(ASSISTANT_ROLE):
            with st.spinner("Processing..."):
                try:
                    # Pass the full message history
                    response_data = rag_engine.process_query(user_input, st.session_state.messages)
                    
                    # Display the answer
                    answer = response_data.get("answer", "I couldn't generate a response.")
                    st.markdown(answer)
                    
                    # Show sources if enabled
                    if st.session_state.get("show_sources", False):
                        source_docs = response_data.get("source_documents", [])
                        if source_docs:
                            with st.expander(f"Retrieved Sources ({len(source_docs)} chunks)"):
                                st.info("Document chunks used to generate the answer:")
                                for i, doc in enumerate(source_docs):
                                    metadata = doc.metadata
                                    source = metadata.get('source', 'Unknown')
                                    page = metadata.get('page', 'N/A')
                                    content_type = metadata.get('content_type', 'text')
                                    
                                    st.text(f"{i+1}. {source} (Page: {page}, Type: {content_type})")
                                    
                                    # Show content preview
                                    content_preview = doc.page_content[:300]
                                    if len(doc.page_content) > 300:
                                        content_preview += "..."
                                    st.text_area(f"Content {i+1}", content_preview, height=100, disabled=True)
                                    st.divider()
                        else:
                            st.warning("No source documents were retrieved for this query.")
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    st.error(f"An error occurred: {str(e)}")
                    answer = "I apologize, but I encountered an error. Please try rephrasing your question."
                    response_data = {"answer": answer, "total_tokens": 0, "total_cost_usd": 0.0}
        
        # Add assistant message
        st.session_state.messages.append({"role": ASSISTANT_ROLE, "content": response_data.get("answer", "Error occurred")})
        
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
                
                if response_data.get("answer"):
                    st.session_state.document_summary = response_data['answer']
                    self._update_statistics(response_data)
                    st.success("Summary generated")
                    st.rerun()
                else:
                    st.error("Failed to generate summary")
                    
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                st.error(f"Failed to generate summary: {e}")
    
    def _update_statistics(self, response_data: Dict[str, Any]):
        """Update usage statistics."""
        tokens = response_data.get("total_tokens", 0)
        cost = response_data.get("total_cost_usd", 0.0)
        source_count = len(response_data.get("source_documents", []))
        
        st.session_state.total_tokens += tokens
        st.session_state.total_cost_usd += cost
        st.session_state.last_query_info = {
            "total_tokens": tokens,
            "total_cost_usd": cost,
            "source_count": source_count
        }
    
    def _get_welcome_message(self) -> str:
        """Get welcome message."""
        return f"""# Welcome to {Config.APP_TITLE}

I'm your technical documentation assistant.

**Capabilities:**
- Technical document analysis
- Code explanation with examples
- Data and table interpretation
- Comprehensive summaries
- Precise answers with citations

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
        
        # Initialize app components
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