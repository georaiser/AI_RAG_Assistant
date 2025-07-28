"""
Application initialization and startup logic.
Support summarization functionality.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional

from src.backend import handle_query, generate_document_summary
from src.data_loader import DocumentLoader
from src.vector_store import initialize_vector_store, load_vector_store
from src.config import config

logger = logging.getLogger(__name__)

class AppLoader:
    """Handles application initialization and loading logic."""
    
    def __init__(self):
        self.loader = DocumentLoader()
        self._initialized = False
    
    def get_document_file(self) -> Optional[Path]:
        """Get the single document file from config."""
        if config.DOCUMENT_PATH.exists() and config.DOCUMENT_PATH.is_file():
            # Check if it's a supported format
            if config.DOCUMENT_PATH.suffix.lower() in config.SUPPORTED_FORMATS:
                return config.DOCUMENT_PATH
        return None
    
    def initialize_app(self) -> bool:
        """Initialize the application with single document processing."""
        if self._initialized:
            return True
            
        try:
            # Check existing vector store
            if config.VECTOR_STORE_PATH.exists() and any(config.VECTOR_STORE_PATH.iterdir()):
                logger.info("Loading existing vector store...")
                vector_store = load_vector_store()
                if vector_store is not None:
                    self._initialized = True
                    return True
            
            # Create new vector store
            logger.info("Creating new vector store...")
            with st.spinner("Processing document..."):
                document_file = self.get_document_file()
                
                if not document_file:
                    st.error("No supported document found.")
                    return False
                
                # Load single document
                try:
                    documents = self.loader.load_and_process_document(document_file)
                    logger.info(f"Loaded {len(documents)} chunks from {document_file.name}")
                except Exception as e:
                    logger.error(f"Error loading {document_file}: {e}")
                    st.error(f"Failed to load {document_file.name}")
                    return False
                
                if not documents:
                    st.error("Failed to load document")
                    return False
                
                # Create vector store
                vector_store = initialize_vector_store(documents)
                if vector_store:
                    st.success("Vector store created successfully!")
                    self._initialized = True
                    return True
                else:
                    st.error("Failed to create vector store")
                    return False
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            st.error(f"Initialization failed: {e}")
            return False
    
    def process_query(self, user_input: str, messages: list) -> dict:
        """Process user query through the backend."""
        return handle_query(user_input, messages)
    
    def generate_summary(self) -> dict:
        """Generate document summary through the backend."""
        return generate_document_summary()
    
    def get_document_info(self) -> dict:
        """Get information about the loaded document."""
        document_file = self.get_document_file()
        if document_file:
            size_kb = document_file.stat().st_size / 1024
            return {
                "name": document_file.name,
                "size_kb": size_kb,
                "type": document_file.suffix.lower(),
                "exists": True
            }
        return {"exists": False}
    
    def is_initialized(self) -> bool:
        """Check if app is initialized."""
        return self._initialized


# Global app loader instance
app_loader = AppLoader()