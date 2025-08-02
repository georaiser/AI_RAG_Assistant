# app.py
"""
Main Streamlit application for RAG Document Assistant.
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
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="📚",
    layout="wide"
)


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_vector_store()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        defaults = {
            "messages": [{"role": "bot", "content": self._get_welcome_message()}],
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "last_query_info": {},
            "vector_store_ready": False
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
                st.info("Initializing vector store for the first time. This may take a few minutes...")
                
                # Load and process documents
                documents = load_data()
                if documents:
                    if vector_manager.initialize_vector_store(documents):
                        st.session_state.vector_store_ready = True
                        st.success("Vector store initialized successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to initialize vector store")
                else:
                    st.error("No documents found to process")
            else:
                st.session_state.vector_store_ready = True
    
    def render_sidebar(self):
        """Render sidebar with statistics and summary option."""
        with st.sidebar:
            st.header("Usage Statistics")
            
            # Last query info
            st.subheader("Last Query")
            if st.session_state.last_query_info:
                info = st.session_state.last_query_info
                st.text(f"Tokens: {info.get('total_tokens', 0)}")
                st.text(f"Cost (USD): ${info.get('total_cost_usd', 0):.6f}")
            else:
                st.text("No queries processed yet")
            
            st.divider()
            
            # Session totals
            st.subheader("Session Total")
            st.text(f"Tokens: {st.session_state.total_tokens}")
            st.text(f"Cost (USD): ${st.session_state.total_cost_usd:.6f}")
            
            st.divider()
            
            # Document summary button
            if st.button("Generate Document Summary", type="primary"):
                self._generate_summary()
    
    def _render_config_options(self):
        """Render configuration options in sidebar."""
        st.subheader("Configuration")
        
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
        
        if st.button("Update Embedding Model"):
            if EmbeddingManager.update_embedding_model(model_type.lower(), model_name):
                st.success("Embedding model updated! Restart app to apply changes.")
        
        # Search strategy
        search_type = st.selectbox(
            "Search Strategy",
            options=Config.AVAILABLE_SEARCH_TYPES,
            index=Config.AVAILABLE_SEARCH_TYPES.index(Config.SEARCH_TYPE)
        )
        
        if search_type != Config.SEARCH_TYPE:
            Config.SEARCH_TYPE = search_type
            st.success(f"Search strategy updated to: {search_type}")
    
    def render_chat_interface(self):
        """Render main chat interface."""
        st.title(Config.APP_TITLE)
        
        # Display chat messages
        for message in st.session_state.messages:
            role = "assistant" if message["role"] == "bot" else message["role"]
            with st.chat_message(role):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            self._process_user_message(prompt)
    
    def _process_user_message(self, user_input: str):
        """Process user message and generate response."""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = self._get_assistant_response(user_input)
                st.markdown(response_data["answer"])
        
        # Add bot message
        st.session_state.messages.append({"role": "bot", "content": response_data["answer"]})
        
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
            return engine.process_query(user_input, st.session_state.messages)
        except Exception as e:
            print(f"ERROR in assistant response: {e}")
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
        
        with st.spinner("Generating document summary..."):
            try:
                engine = RAGEngine()
                response_data = engine.generate_summary()
                
                # Add summary to chat
                summary_message = f"**Document Summary:**\n\n{response_data['answer']}"
                st.session_state.messages.append({
                    "role": "bot", 
                    "content": summary_message
                })
                
                # Update statistics
                self._update_statistics(response_data)
                
                st.success("Document summary generated!")
                st.rerun()
                
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
        """Get welcome message for the application."""
        doc_mode = "multiple documents" if Config.PROCESS_MULTIPLE_DOCUMENTS else "single document"
        
        return f"""Welcome to the **{Config.APP_TITLE}**!

I'm here to help you explore and understand your technical documents. I can:

- Answer specific questions about document content
- Provide detailed explanations with proper citations
- Extract information from tables and figures
- Generate comprehensive document summaries

**Current Configuration:**
- Processing mode: {doc_mode}
- Embedding model: {Config.OPENAI_EMBEDDING_MODEL if Config.EMBEDDING_TYPE == 'openai' else Config.HF_EMBEDDING_MODEL}
- Search strategy: {Config.SEARCH_TYPE}

All my responses include proper source citations. Use the sidebar to generate a document summary or adjust settings.

How can I help you today?"""

    def run(self):
        """Main application entry point."""
        # Validate configuration
        if not Config.validate():
            st.error("Configuration validation failed. Please check your settings.")
            return
        
        # Setup directories
        if not Config.setup_directories():
            st.error("Failed to setup required directories.")
            return
        
        # Render interface
        self.render_sidebar()
        self.render_chat_interface()


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()