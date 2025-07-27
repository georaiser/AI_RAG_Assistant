"""
Streamlit app for Docu Bot - UI components only.
"""

import streamlit as st
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy logs
for lib in ["chromadb", "httpx", "openai", "langchain"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

from app_loader import app_loader
from src.data_loader import DocumentLoaderFactory
from src.config import config

# Page config
st.set_page_config(
    page_title="Docu Bot",
    page_icon="📚🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_sidebar() -> None:
    """Render sidebar with statistics and document info."""
    with st.sidebar:
        st.title("Statistics")
        
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
        st.subheader("Document")
        doc_info = app_loader.get_document_info()
        
        if doc_info["exists"]:
            st.metric("File", "1")
            with st.expander("Document Details"):
                st.write(f"**Name:** {doc_info['name']}")
                st.write(f"**Size:** {doc_info['size_kb']:.1f} KB")
                st.write(f"**Type:** {doc_info['type']}")
        else:
            st.error("No document found")
        
        # Reset button
        if st.button("🔄 Reset Stats", type="secondary"):
            for key in ['query_count', 'total_tokens', 'total_cost', 'last_query_info']:
                st.session_state[key] = 0 if key != 'last_query_info' else {}
            st.rerun()

def initialize_session_state() -> None:
    """Initialize session state with defaults."""
    if "messages" not in st.session_state:
        doc_info = app_loader.get_document_info()
        supported = ", ".join(DocumentLoaderFactory.supported_extensions())
        
        doc_status = f"**Document:** {doc_info['name']}" if doc_info["exists"] else "** Document:** Not found"
        
        welcome = f"""**Welcome to DocuPy Bot!** 🤖

I'm your document assistant powered by advanced RAG.

{doc_status}
**Supported formats:** {supported}

**What I can help with:**
- Answer questions about your document
- Find specific information quickly  
- Summarize content and extract key points
- Provide contextual explanations

Ask me anything about your document!"""

        st.session_state.messages = [{"role": "assistant", "content": welcome}]
    
    # Initialize counters
    for key, default in [
        ("query_count", 0), ("total_tokens", 0), 
        ("total_cost", 0.0), ("app_initialized", False)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

def render_chat_interface():
    """Render the main chat interface."""
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                try:
                    # Handle query
                    result = app_loader.process_query(user_input, st.session_state.messages)
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
                        with st.expander(f"Sources ({len(sources)})"):
                            for source in sources:
                                st.caption(f"• {source}")
                    
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Query error: {e}")
                    error_msg = "Error processing query. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Header
    st.title("Docu Bot")
    st.caption("*Intelligent Document Assistant with RAG*")
    
    # Initialize app once
    if not st.session_state.app_initialized:
        if app_loader.initialize_app():
            st.session_state.app_initialized = True
            st.rerun()
        else:
            st.stop()
    
    # Sidebar
    render_sidebar()
    
    # Main chat interface
    render_chat_interface()

if __name__ == "__main__":
    main()