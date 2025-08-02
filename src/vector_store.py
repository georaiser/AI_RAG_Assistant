# vector_store.py
"""
Vector store management for RAG system.
Handles embedding creation and retrieval operations.
"""

from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import Config


class VectorStoreManager:
    """Manages vector store operations for document retrieval."""
    
    def __init__(self):
        self.embeddings = self._create_embeddings()
        self.vector_store = None
    
    def _create_embeddings(self):
        """Create embedding model based on configuration."""
        if Config.EMBEDDING_TYPE == "openai":
            return OpenAIEmbeddings(
                model=Config.OPENAI_EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=Config.HF_EMBEDDING_MODEL
            )
    
    def initialize_vector_store(self, documents: List[Document]) -> bool:
        """Initialize vector store with documents."""
        if not documents:
            print("ERROR: No documents provided for vector store initialization")
            return False
        
        try:
            # Ensure directory exists
            Config.VECTOR_STORE_PATH.mkdir(exist_ok=True)
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(Config.VECTOR_STORE_PATH)
            )
            
            print(f"Vector store initialized with {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"ERROR initializing vector store: {e}")
            return False
    
    def load_vector_store(self) -> Optional[Chroma]:
        """Load existing vector store."""
        try:
            if not Config.VECTOR_STORE_PATH.exists():
                print("Vector store directory does not exist")
                return None
            
            self.vector_store = Chroma(
                persist_directory=str(Config.VECTOR_STORE_PATH),
                embedding_function=self.embeddings
            )
            
            print("Vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            print(f"ERROR loading vector store: {e}")
            return None
    
    def get_retriever(self):
        """Get retriever with configured search parameters."""
        if not self.vector_store:
            return None
        
        search_kwargs = {"k": Config.RETRIEVAL_K}
        
        if Config.SEARCH_TYPE == "mmr":
            search_kwargs.update({
                "fetch_k": Config.FETCH_K,
                "lambda_mult": Config.LAMBDA_MULT
            })
        elif Config.SEARCH_TYPE == "similarity_score_threshold":
            search_kwargs["score_threshold"] = Config.SCORE_THRESHOLD
        
        return self.vector_store.as_retriever(
            search_type=Config.SEARCH_TYPE,
            search_kwargs=search_kwargs
        )


# Convenience functions for backward compatibility
def initialize_vector_store(documents: List[Document]) -> bool:
    """Initialize vector store with documents."""
    manager = VectorStoreManager()
    return manager.initialize_vector_store(documents)


def load_vector_store() -> Optional[Chroma]:
    """Load existing vector store."""
    manager = VectorStoreManager()
    return manager.load_vector_store()