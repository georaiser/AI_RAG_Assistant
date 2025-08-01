"""
Application initialization and startup logic.
Support summarization functionality.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional

from src.backend import handle_query, generate_document_summary, set_vector_manager
from src.data_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.config import config

logger = logging.getLogger(__name__)

class AppLoader:
    """Handles application initialization and loading logic."""
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.vector_manager = VectorStoreManager()
        self._initialized = False
        self._document_info = None
    
    def get_document_file(self) -> Optional[Path]:
        """Get the single document file from config."""
        if config.DOCUMENT_PATH.exists() and config.DOCUMENT_PATH.is_file():
            if config.DOCUMENT_PATH.suffix.lower() in config.SUPPORTED_FORMATS:
                return config.DOCUMENT_PATH
        return None
    
    def initialize_app(self) -> bool:
        """Initialize the application with single document processing."""
        if self._initialized:
            return True
            
        try:
            # Check existing vector store first
            if self._check_existing_vector_store():
                logger.info("Using existing vector store")
                # Set the vector manager in backend
                set_vector_manager(self.vector_manager)
                self._initialized = True
                return True
            
            # Create new vector store
            success = self._create_new_vector_store()
            if success:
                # Set the vector manager in backend
                set_vector_manager(self.vector_manager)
                self._initialized = True
            return success
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            st.error(f"Initialization failed: {str(e)}")
            return False
    
    def _check_existing_vector_store(self) -> bool:
        """Check and load existing vector store."""
        if config.VECTOR_STORE_PATH.exists() and any(config.VECTOR_STORE_PATH.iterdir()):
            loaded_store = self.vector_manager.load()
            if loaded_store and self.vector_manager.is_initialized():
                return True
        return False
    
    def _create_new_vector_store(self) -> bool:
        """Create new vector store from document."""
        logger.info("Creating new vector store...")
        with st.spinner("Processing document..."):
            document_file = self.get_document_file()
            
            if not document_file:
                st.error("No supported document found. Please check the document path in config.")
                return False
            
            # Load and process document
            documents = self.loader.load_and_process_document(document_file)
            logger.info(f"Successfully loaded {len(documents)} chunks from {document_file.name}")
            
            if not documents:
                st.error(f"No content could be extracted from {document_file.name}")
                return False
            
            # Create vector store
            vector_store = self.vector_manager.initialize(documents)
            if vector_store and self.vector_manager.is_initialized():
                st.success(f"Vector store created successfully with {len(documents)} chunks!")
                return True
            else:
                st.error("Failed to create vector store")
                return False
    
    def process_query(self, user_input: str, messages: list) -> dict:
        """Process user query through the backend."""
        if not self._initialized or not self.vector_manager.is_initialized():
            return {
                "answer": "System not initialized. Please restart the application.",
                "sources": [],
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "documents_retrieved": 0
            }
        
        return handle_query(user_input, messages)
    
    def generate_summary(self) -> dict:
        """Generate document summary through the backend."""
        if not self._initialized or not self.vector_manager.is_initialized():
            return {
                "summary": "System not initialized. Please restart the application.",
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "documents_used": 0
            }
        
        return generate_document_summary()
    
    def get_document_info(self) -> dict:
        """Get information about the loaded document."""
        if self._document_info:
            return self._document_info
        
        document_file = self.get_document_file()
        if document_file:
            size_kb = document_file.stat().st_size / 1024
            self._document_info = {
                "name": document_file.name,
                "size_kb": size_kb,
                "type": document_file.suffix.lower(),
                "path": str(document_file),
                "exists": True
            }
        else:
            self._document_info = {
                "exists": False, 
                "error": f"Document not found at {config.DOCUMENT_PATH}"
            }
        
        return self._document_info
    
    def is_initialized(self) -> bool:
        """Check if app is initialized."""
        return self._initialized and self.vector_manager.is_initialized()
    
    def reset(self) -> None:
        """Reset the application state."""
        self._initialized = False
        self._document_info = None
        self.vector_manager.reset()


# Global app loader instance
app_loader = AppLoader()