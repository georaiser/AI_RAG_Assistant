"""
Streamlit application for the RAG system.
Clean, simple interface that uses the main RAG system class.
"""

import streamlit as st
import logging
from rag_system import RAGSystem
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Streamlit page
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide"
)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching for performance."""
    try:
        rag = RAGSystem()
        if rag.initialize():
            return rag
        else:
            st.error("Failed to initialize RAG system")
            return None
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "messages": [{
            "role": "assistant",  # Changed from "bot" to "assistant" for consistency
            "content": f"""Hello! I'm **{config.BOT_NAME}**.

I'm a specialized assistant for {config.BOT_DESCRIPTION}. My knowledge base contains the loaded documentation.

**You can ask me about:**
* Syntax and usage of standard modules
* Function and class explanations  
* Code examples from the documentation

Simply type your question and I'll search for the most relevant information. How can I help you?"""
        }],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "last_query_info": {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_statistics():
    """Display usage statistics in sidebar."""
    with st.sidebar:
        st.title("📊 Usage Statistics")
        st.markdown("---")
        
        st.subheader("Last Query")
        if st.session_state.last_query_info:
            info = st.session_state.last_query_info
            st.text(f"Tokens: {info.get('total_tokens', 0)}")
            st.text(f"Cost (USD): ${info.get('total_cost_usd', 0):.6f}")
        else:
            st.text("No queries processed yet.")
        
        st.markdown("---")
        st.subheader("Session Total")
        st.text(f"Tokens: {st.session_state.total_tokens}")
        st.text(f"Cost (USD): ${st.session_state.total_cost_usd:.6f}")

def update_statistics(result_data):
    """Update session statistics with query results."""
    st.session_state.total_tokens += result_data.get("total_tokens", 0)
    st.session_state.total_cost_usd += result_data.get("total_cost_usd", 0)
    st.session_state.last_query_info = {
        "total_tokens": result_data.get("total_tokens", 0),
        "total_cost_usd": result_data.get("total_cost_usd", 0)
    }

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    if not rag_system:
        st.stop()
    
    # Display title and statistics
    st.title(f"{config.APP_TITLE} {config.APP_ICON}")
    display_statistics()
    
    # Chat input
    user_input = st.chat_input(f"Ask about {config.BOT_DESCRIPTION}...")
    
    # Process user input
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response from RAG system
        with st.spinner("Thinking..."):
            result_data = rag_system.query(user_input, st.session_state.messages)
        
        # Add bot response
        st.session_state.messages.append({
            "role": "assistant",  # Changed from "bot" to "assistant" for consistency
            "content": result_data["answer"]
        })
        
        # Update statistics
        update_statistics(result_data)
        
        # Rerun to update sidebar
        st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        # Map "assistant" to "assistant" and "user" to "user" for st.chat_message
        with st.chat_message(role):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()