"""
Vector store.
"""

import logging
from typing import List, Optional
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from pydantic import SecretStr

from src.config import config

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store manager."""
    
    def __init__(self):
        self.persist_directory = config.VECTOR_STORE_PATH
        self.embeddings = self._get_embeddings()
        self._vector_store: Optional[Chroma] = None
    
    def _get_embeddings(self):
        """Get embedding model based on config."""
        if config.EMBEDDING_TYPE == "huggingface":
            logger.info(f"Using HuggingFace embeddings: {config.HF_EMBEDDING_MODEL}")
            return HuggingFaceEmbeddings(
                model_name=config.HF_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}, # Use 'cuda' for GPU
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            logger.info(f"Using OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL}")
            return OpenAIEmbeddings(
                api_key=SecretStr(config.OPENAI_API_KEY),
                model=config.OPENAI_EMBEDDING_MODEL
            )
    
    def initialize(self, documents: List[Document]) -> Optional[Chroma]:
        """Initialize vector store with documents."""
        if not documents:
            logger.error("No documents provided")
            return None
        
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating vector store with {len(documents)} documents")
            self._vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            logger.info("Vector store initialized successfully")
            return self._vector_store
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            return None
    
    def load(self) -> Optional[Chroma]:
        """Load existing vector store."""
        try:
            if not self._exists():
                logger.warning("Vector store doesn't exist")
                return None
            
            self._vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            
            logger.info("Vector store loaded successfully")
            return self._vector_store
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None
    
    def get_retriever(self, k: Optional[int] = None):
        """Get retriever with optimized search configuration."""
        if not self._vector_store:
            logger.error("Vector store not initialized")
            return None
        
        k = k or config.RETRIEVAL_K
        
        # Configure retriever based on search type
        if config.SEARCH_TYPE == "mmr":
            return self._vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": config.FETCH_K,
                    "lambda_mult": config.LAMBDA_MULT
                }
            )
        elif config.SEARCH_TYPE == "similarity_score_threshold":
            return self._vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": config.SCORE_THRESHOLD
                }
            )
        else:
            # Standard similarity search (fastest)
            return self._vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
    
    def _exists(self) -> bool:
        """Check if vector store exists."""
        return (self.persist_directory.exists() and 
                self.persist_directory.is_dir() and 
                any(self.persist_directory.iterdir()))


# Global instance
_vector_store = VectorStore()

def initialize_vector_store(documents: List[Document]) -> Optional[Chroma]:
    """Initialize global vector store."""
    return _vector_store.initialize(documents)

def load_vector_store() -> Optional[Chroma]:
    """Load existing vector store."""
    return _vector_store.load()

def get_retriever(k: Optional[int] = None):
    """Get retriever from global vector store."""
    return _vector_store.get_retriever(k)