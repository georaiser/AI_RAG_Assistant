"""
Main RAG system class that orchestrates all components.
Provides a simple interface for document loading, indexing, and querying.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from data_loader import DocumentLoader
from vector_store import VectorStoreManager
from backend import QueryProcessor
from config import config

class RAGSystem:
    """
    Main RAG system that coordinates all components.
    Follows single responsibility principle with clear separation of concerns.
    """
    
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.vector_store_manager = VectorStoreManager()
        self.query_processor = QueryProcessor(self.vector_store_manager)
        self._initialized = False
    
    def initialize(self, document_path: Optional[Path] = None) -> bool:
        """
        Initialize the RAG system.
        
        Args:
            document_path: Path to document file. If None, uses config default.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print("=== RAG System Initialization ===")
            
            # Try to load existing vector store first
            if self.vector_store_manager.load_vector_store():
                print("Existing vector store loaded successfully")
                self._initialized = True
                return True
            
            # If no existing store, create new one
            print("No existing vector store found. Creating new one...")
            
            # Load and process documents
            documents = self.document_loader.load_documents(document_path)
            if not documents:
                print("ERROR: No documents loaded")
                return False
            
            # Initialize vector store
            if not self.vector_store_manager.initialize_vector_store(documents):
                print("ERROR: Failed to initialize vector store")
                return False
            
            self._initialized = True
            print("RAG system initialized successfully")
            return True
            
        except Exception as e:
            print(f"ERROR during RAG system initialization: {e}")
            return False
    
    def query(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            question: User question
            chat_history: Previous conversation messages
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self._initialized:
            return {
                "answer": "System not initialized. Please initialize first.",
                "source_documents": [],
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0.0,
                "error": "System not initialized"
            }
        
        chat_history = chat_history or []
        return self.query_processor.process_query(question, chat_history)
    
    def add_documents(self, document_path: Path) -> bool:
        """
        Add new documents to existing vector store.
        
        Args:
            document_path: Path to new document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            documents = self.document_loader.load_documents(document_path)
            if not documents:
                return False
            
            # Reinitialize with new documents
            return self.vector_store_manager.initialize_vector_store(documents)
            
        except Exception as e:
            print(f"ERROR adding documents: {e}")
            return False
    
    def search_similar(self, query: str, k: Optional[int] = None) -> List[str]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of similar document contents
        """
        if not self._initialized:
            return []
        
        k_value = k if k is not None else config.RETRIEVAL_K
        documents = self.vector_store_manager.similarity_search(query, k_value)
        return [doc.page_content for doc in documents]
    
    def is_ready(self) -> bool:
        """Check if system is ready to process queries."""
        return self._initialized and self.vector_store_manager.is_initialized()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system configuration and status information."""
        return {
            "initialized": self._initialized,
            "vector_store_ready": self.vector_store_manager.is_initialized(),
            "model_name": config.MODEL_NAME,
            "chunk_size": config.CHUNK_SIZE,
            "retrieval_k": config.RETRIEVAL_K,
            "app_title": config.APP_TITLE,
            "bot_name": config.BOT_NAME
        }