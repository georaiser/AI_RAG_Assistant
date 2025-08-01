"""
Vector store with simplified management and better retrieval configuration.
"""

import logging
from typing import List, Optional
import shutil

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from pydantic import SecretStr

from src.config import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Simplified vector store manager with better retrieval."""
    
    def __init__(self):
        self.persist_directory = config.VECTOR_STORE_PATH
        self.embeddings = None
        self._vector_store: Optional[Chroma] = None
        self._initialized = False
    
    def _get_embeddings(self):
        """Get embedding model based on config."""
        if self.embeddings is not None:
            return self.embeddings
            
        if config.EMBEDDING_TYPE == "huggingface":
            logger.info(f"Initializing HuggingFace embeddings: {config.HF_EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.HF_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            logger.info(f"Initializing OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL}")
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
                
            self.embeddings = OpenAIEmbeddings(
                api_key=SecretStr(config.OPENAI_API_KEY),
                model=config.OPENAI_EMBEDDING_MODEL
            )
                
        return self.embeddings
    
    def initialize(self, documents: List[Document]) -> Optional[Chroma]:
        """Initialize vector store with documents."""
        if not documents:
            logger.error("No documents provided for vector store initialization")
            return None
        
        try:
            # Validate documents
            valid_docs = self._validate_documents(documents)
            if not valid_docs:
                logger.error("No valid documents after validation")
                return None
            
            # Get embeddings
            embeddings = self._get_embeddings()
            
            # Prepare directory
            self._prepare_directory()
            
            logger.info(f"Creating vector store with {len(valid_docs)} valid documents")
            
            # Create vector store
            self._vector_store = Chroma.from_documents(
                documents=valid_docs,
                embedding=embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            # Simple verification
            if self._vector_store and self._vector_store._collection:
                logger.info("Vector store created successfully")
                self._initialized = True
                return self._vector_store
            else:
                logger.error("Vector store creation failed")
                return None
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            self._cleanup_failed_initialization()
            return None
    
    def load(self) -> Optional[Chroma]:
        """Load existing vector store."""
        try:
            if not self._exists():
                return None
            
            # Get embeddings
            embeddings = self._get_embeddings()
            
            logger.info("Loading existing vector store...")
            self._vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=embeddings
            )
            
            # Simple verification
            if self._vector_store and self._vector_store._collection:
                logger.info("Vector store loaded successfully")
                self._initialized = True
                return self._vector_store
            else:
                return None
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None
    
    def get_retriever(self, k: Optional[int] = None):
        """Get retriever with configured search type."""
        if not self._vector_store or not self._initialized:
            logger.error("Vector store not initialized")
            return None
        
        k = k or config.RETRIEVAL_K
        
        try:
            # Configure retriever based on search type
            if config.SEARCH_TYPE == "mmr":
                logger.debug(f"Creating MMR retriever with k={k}")
                return self._vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "fetch_k": config.FETCH_K,
                        "lambda_mult": config.LAMBDA_MULT
                    }
                )
            elif config.SEARCH_TYPE == "similarity_score_threshold":
                logger.debug(f"Creating similarity threshold retriever with k={k}")
                return self._vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": k,
                        "score_threshold": config.SCORE_THRESHOLD
                    }
                )
            else:
                # Standard similarity search
                logger.debug(f"Creating similarity retriever with k={k}")
                return self._vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
                
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            return None
    
    def is_initialized(self) -> bool:
        """Check if vector store is properly initialized."""
        return self._initialized and self._vector_store is not None
    
    def _validate_documents(self, documents: List[Document]) -> List[Document]:
        """Validate and filter documents."""
        valid_docs = []
        
        for doc in documents:
            # Check content
            if not doc.page_content or not doc.page_content.strip():
                continue
            
            # Check metadata
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {"page": 1, "source": "unknown"}
            
            # Ensure required metadata fields
            if 'page' not in doc.metadata:
                doc.metadata['page'] = 1
            if 'source' not in doc.metadata:
                doc.metadata['source'] = 'unknown'
            
            valid_docs.append(doc)
        
        logger.info(f"Validated {len(valid_docs)} out of {len(documents)} documents")
        return valid_docs
    
    def _prepare_directory(self) -> None:
        """Prepare vector store directory."""
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
    
    def _cleanup_failed_initialization(self) -> None:
        """Clean up after failed initialization."""
        try:
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")
    
    def _exists(self) -> bool:
        """Check if vector store exists and has content."""
        try:
            return (self.persist_directory.exists() and 
                    self.persist_directory.is_dir() and 
                    any(self.persist_directory.iterdir()))
        except Exception:
            return False

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        if not self._vector_store or not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            collection = self._vector_store._collection
            count = collection.count() if collection else 0
            
            return {
                "status": "initialized",
                "document_count": count,
                "embedding_type": config.EMBEDDING_TYPE,
                "search_type": config.SEARCH_TYPE,
                "retrieval_k": config.RETRIEVAL_K
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def reset(self) -> bool:
        """Reset vector store by removing all data."""
        try:
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
            
            self._vector_store = None
            self._initialized = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")
            return False


# Global instance
_vector_store_manager = VectorStoreManager()

def initialize_vector_store(documents: List[Document]) -> Optional[Chroma]:
    """Initialize global vector store."""
    return _vector_store_manager.initialize(documents)

def load_vector_store() -> Optional[Chroma]:
    """Load existing vector store."""
    return _vector_store_manager.load()

def get_retriever(k: Optional[int] = None):
    """Get retriever from global vector store."""
    return _vector_store_manager.get_retriever(k)

def get_vector_store_stats() -> dict:
    """Get vector store statistics."""
    return _vector_store_manager.get_stats()

def reset_vector_store() -> bool:
    """Reset the global vector store."""
    return _vector_store_manager.reset()