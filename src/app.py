"""
Clean Streamlit app for Docu Bot.
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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_sidebar() -> None:
    """Render clean sidebar with statistics, document info, and summary feature."""
    with st.sidebar:
        st.title("📊 Statistics")
        
        # Session stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.get('query_count', 0))
            st.metric("Total Tokens", st.session_state.get('total_tokens', 0))
        with col2:
            st.metric("Cost (USD)", f"${st.session_state.get('total_cost', 0):.4f}")
            
        # Last query info
        if last_info := st.session_state.get('last_query_info'):
            with st.expander("📝 Last Query Details"):
                st.write(f"**Tokens:** {last_info.get('total_tokens', 0)}")
                st.write(f"**Cost:** ${last_info.get('total_cost_usd', 0):.6f}")
                st.write(f"**Documents:** {last_info.get('documents_retrieved', 0)}")
                
                if sources := last_info.get('sources'):
                    st.write(f"**Sources:** {len(sources)}")
                    for source in sources[:3]:  # Show max 3
                        st.caption(source)
        
        st.divider()
        
        # Document info
        st.subheader("📄 Document")
        doc_info = app_loader.get_document_info()
        
        if doc_info["exists"]:
            st.metric("File", "1")
            with st.expander("Document Details"):
                st.write(f"**Name:** {doc_info['name']}")
                st.write(f"**Size:** {doc_info['size_kb']:.1f} KB")
                st.write(f"**Type:** {doc_info['type']}")
        else:
            st.error("No document found")
        
        st.divider()
        
        # Document Summary Section
        st.subheader("📋 Summary")
        
        # Check if summary exists in session state
        if 'document_summary' not in st.session_state:
            st.session_state.document_summary = None
        
        # Summary button
        if st.button("🔍 Generate Summary", type="primary", use_container_width=True):
            if doc_info["exists"]:
                with st.spinner("Generating summary..."):
                    try:
                        from src.backend import generate_document_summary
                        summary_result = generate_document_summary()
                        
                        if summary_result and summary_result.get('summary'):
                            st.session_state.document_summary = summary_result
                            
                            # Update total costs
                            st.session_state.total_tokens += summary_result.get('total_tokens', 0)
                            st.session_state.total_cost += summary_result.get('total_cost_usd', 0.0)
                            
                            st.success("Summary generated!")
                            st.rerun()
                        else:
                            st.error("Failed to generate summary")
                    except Exception as e:
                        logger.error(f"Summary generation error: {e}")
                        st.error("Error generating summary. Please try again.")
            else:
                st.error("No document available for summary")
        
        # Display summary if available
        if st.session_state.document_summary:
            summary_data = st.session_state.document_summary
            
            with st.expander("📋 Document Summary", expanded=True):
                st.write(summary_data.get('summary', 'No summary available'))
                
                # Summary stats
                if summary_data.get('total_tokens', 0) > 0:
                    st.caption(f"📊 Tokens: {summary_data.get('total_tokens', 0)} | "
                             f"💰 Cost: ${summary_data.get('total_cost_usd', 0):.4f} | "
                             f"📑 Sources: {summary_data.get('documents_used', 0)}")
            
            # Clear summary button
            if st.button("🗑️ Clear Summary", type="secondary", use_container_width=True):
                st.session_state.document_summary = None
                st.rerun()
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset Stats", type="secondary", use_container_width=True):
            keys_to_reset = [
                'query_count', 'total_tokens', 'total_cost', 
                'last_query_info', 'document_summary'
            ]
            for key in keys_to_reset:
                if key in ['query_count', 'total_tokens', 'total_cost']:
                    st.session_state[key] = 0
                else:
                    st.session_state[key] = None if key == 'document_summary' else {}
            st.success("Statistics reset!")
            st.rerun()

def initialize_session_state() -> None:
    """Initialize session state with defaults."""
    if "messages" not in st.session_state:
        doc_info = app_loader.get_document_info()
        supported = ", ".join(DocumentLoaderFactory.supported_extensions())
        
        doc_status = f"**Document:** {doc_info['name']}" if doc_info["exists"] else "**Document:** Not found"
        
        welcome = f"""**Welcome to Docu Bot!** 🤖

I'm your intelligent document assistant powered by advanced RAG technology.

{doc_status}

**Supported formats:** {supported}

**What I can help with:**
- Answer detailed questions about your document
- Find specific information and examples
- Explain complex concepts and procedures
- Provide summaries and key insights
- Reference specific sections and pages

💡 **Tip:** Ask specific questions for the best results. I can handle follow-up questions and remember our conversation context.

Ask me anything about your document!"""

        st.session_state.messages = [{"role": "assistant", "content": welcome}]
    
    # Initialize counters and states
    defaults = {
        "query_count": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "app_initialized": False,
        "document_summary": None,
        "last_query_info": {}
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def render_chat_interface():
    """Render the main chat interface with enhanced user experience."""
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask about your document...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing document..."):
                try:
                    # Handle query with current messages
                    result = app_loader.process_query(user_input, st.session_state.messages)
                    response = result.get("answer", "I couldn't process your query.")
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to messages
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Update session stats
                    st.session_state.query_count += 1
                    st.session_state.total_tokens += result.get("total_tokens", 0)
                    st.session_state.total_cost += result.get("total_cost_usd", 0.0)
                    st.session_state.last_query_info = result
                    
                    # Show sources if available
                    if sources := result.get("sources"):
                        with st.expander(f"📚 Sources ({len(sources)})"):
                            for i, source in enumerate(sources, 1):
                                st.caption(f"{i}. {source}")
                    
                    # Show additional info if helpful
                    docs_retrieved = result.get("documents_retrieved", 0)
                    if docs_retrieved > 0:
                        st.caption(f"ℹ️ Found information from {docs_retrieved} document sections")
                    
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Query error: {e}")
                    error_msg = "I apologize, but I encountered an error processing your question. Please try rephrasing it or ask about a different aspect of the document."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Header
    st.title("📚 Docu Bot")
    st.caption("*Intelligent Document Assistant powered by Advanced RAG*")
    
    # Initialize app once
    if not st.session_state.app_initialized:
        with st.spinner("🚀 Initializing document processing..."):
            if app_loader.initialize_app():
                st.session_state.app_initialized = True
                st.success("✅ Document loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to initialize. Please check your document and configuration.")
                st.stop()
    
    # Sidebar
    render_sidebar()
    
    # Main chat interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.caption("🤖 Powered by OpenAI GPT-3.5 Turbo | Built with Streamlit & LangChain")

if __name__ == "__main__":
    main()