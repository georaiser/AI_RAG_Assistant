"""
Application initialization and startup logic.
Support summarization functionality.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional
from typing import Dict, List, Any

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
    
    def get_document_files(self) -> List[Path]:
        """Get all supported document files."""
        documents = []
        
        # Check documents directory
        docs_dir = config.DOCUMENTS_DIR
        if docs_dir.exists():
            for file_path in docs_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in config.SUPPORTED_FORMATS:
                    documents.append(file_path)
        
        # Also check single document path for backward compatibility
        if config.DOCUMENT_PATH.exists() and config.DOCUMENT_PATH.is_file():
            if config.DOCUMENT_PATH.suffix.lower() in config.SUPPORTED_FORMATS:
                documents.append(config.DOCUMENT_PATH)
        
        return documents[:config.MAX_DOCUMENTS]  # Limit number of documents
    
    def initialize_app(self) -> bool:
        """Initialize with multiple documents."""
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
            
            # Load multiple documents
            document_files = self.get_document_files()
            if not document_files:
                st.error("No supported documents found.")
                return False
            
            with st.spinner(f"Processing {len(document_files)} documents..."):
                all_documents = self.loader.load_multiple_documents(document_files)
                
                if not all_documents:
                    st.error("Failed to load documents")
                    return False
                
                # Create vector store
                vector_store = initialize_vector_store(all_documents)
                if vector_store:
                    st.success(f"Processed {len(document_files)} documents successfully!")
                    self._initialized = True
                    return True
                else:
                    st.error("Failed to create vector store")
                    return False
        
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            st.error(f"Initialization failed: {e}")
            return False
    
    def get_documents_info(self) -> Dict[str, Any]:
        """Get information about all loaded documents."""
        document_files = self.get_document_files()
        total_size = sum(f.stat().st_size for f in document_files) / 1024  # KB
        
        documents_info = []
        for doc_file in document_files:
            size_kb = doc_file.stat().st_size / 1024
            documents_info.append({
                "name": doc_file.name,
                "size_kb": size_kb,
                "type": doc_file.suffix.lower(),
                "path": str(doc_file)
            })
        
        return {
            "count": len(document_files),
            "total_size_kb": total_size,
            "documents": documents_info,
            "exists": len(document_files) > 0
        }

    def process_query(self, user_input: str, messages: list) -> dict:
        """Process user query through the backend."""
        return handle_query(user_input, messages)
    
    def generate_summary(self) -> dict:
        """Generate document summary through the backend."""
        return generate_document_summary()
        
    def is_initialized(self) -> bool:
        """Check if app is initialized."""
        return self._initialized


# Global app loader instance
app_loader = AppLoader()