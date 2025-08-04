# vector_store.py
"""
Vector store management with improved error handling and debugging.
"""

import logging
import shutil
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import Config
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Vector store manager with improved error handling."""
    
    def __init__(self):
        """Initialize with embeddings."""
        self.embeddings = None
        self.vector_store = None
        self._create_embeddings()
    
    def _create_embeddings(self):
        """Create embedding model with better error handling."""
        try:
            if Config.EMBEDDING_TYPE == "openai":
                if not Config.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key required for OpenAI embeddings")
                
                self.embeddings = OpenAIEmbeddings(
                    model=Config.OPENAI_EMBEDDING_MODEL,
                    api_key=SecretStr(Config.OPENAI_API_KEY)
                )
                logger.info(f"Created OpenAI embeddings: {Config.OPENAI_EMBEDDING_MODEL}")
            else:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=Config.HF_EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"Created HuggingFace embeddings: {Config.HF_EMBEDDING_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    
    def initialize_vector_store(self, documents: List[Document]) -> bool:
        """Create new vector store with better validation."""
        if not documents:
            logger.error("No documents provided")
            return False
        
        try:
            Config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
            
            # Filter valid documents
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            
            if not valid_docs:
                logger.error("No valid documents with content")
                return False
            
            if Config.DEBUG_MODE:
                logger.info(f"Creating vector store with {len(valid_docs)} documents")
                # Log sample of document sources
                sources = set(doc.metadata.get('source', 'Unknown') for doc in valid_docs[:10])
                logger.info(f"Sample sources: {list(sources)}")
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=valid_docs,
                embedding=self.embeddings,
                persist_directory=str(Config.VECTOR_STORE_PATH)
            )
            
            # Verify creation
            count = self._get_safe_count()
            if count == 0:
                logger.error("Vector store created but has no documents")
                return False
            
            logger.info(f"Vector store created successfully with {count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            self.vector_store = None
            return False
    
    def load_vector_store(self) -> bool:
        """Load existing vector store with improved validation."""
        try:
            # Check if directory exists
            if not Config.VECTOR_STORE_PATH.exists():
                if Config.DEBUG_MODE:
                    logger.info("Vector store directory doesn't exist")
                return False
            
            # Check for Chroma files
            chroma_files = (
                list(Config.VECTOR_STORE_PATH.glob("*.parquet")) + 
                list(Config.VECTOR_STORE_PATH.glob("chroma.sqlite3")) +
                list(Config.VECTOR_STORE_PATH.glob("*.db"))
            )
            
            if not chroma_files:
                if Config.DEBUG_MODE:
                    logger.info("No Chroma database files found")
                return False
            
            # Load vector store
            self.vector_store = Chroma(
                persist_directory=str(Config.VECTOR_STORE_PATH),
                embedding_function=self.embeddings
            )
            
            # Verify it has documents
            count = self._get_safe_count()
            if count == 0:
                logger.warning("Vector store loaded but appears empty")
                return False
            
            logger.info(f"Vector store loaded with {count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.vector_store = None
            return False
    
    def _get_safe_count(self) -> int:
        """Safely get document count."""
        try:
            if self.vector_store and hasattr(self.vector_store, '_collection'):
                return self.vector_store._collection.count()
            return 0
        except Exception as e:
            if Config.DEBUG_MODE:
                logger.warning(f"Could not get document count: {e}")
            return 0
    
    def is_ready(self) -> bool:
        """Check if vector store is ready."""
        ready = self.vector_store is not None and self._get_safe_count() > 0
        if Config.VERBOSE_MODE:
            logger.info(f"Vector store ready: {ready}")
        return ready
    
    def get_retriever(self, search_type: Optional[str] = None, search_kwargs: Optional[Dict] = None):
        """Get retriever with improved configuration."""
        if not self.is_ready():
            logger.error("Vector store not ready")
            return None
        
        try:
            # Use config defaults if not specified
            final_search_type = search_type or Config.SEARCH_TYPE
            
            if search_kwargs is None:
                search_kwargs = {"k": Config.RETRIEVAL_K}
                
                if final_search_type == "mmr":
                    search_kwargs.update({
                        "fetch_k": Config.FETCH_K,
                        "lambda_mult": Config.LAMBDA_MULT
                    })
                elif final_search_type == "similarity_score_threshold":
                    search_kwargs["score_threshold"] = Config.SCORE_THRESHOLD
            
            retriever = self.vector_store.as_retriever(
                search_type=final_search_type,
                search_kwargs=search_kwargs
            )
            
            if Config.DEBUG_MODE:
                logger.info(f"Created retriever: {final_search_type}, k={search_kwargs.get('k')}")
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            return None
    
    def get_document_count(self) -> int:
        """Get document count."""
        return self._get_safe_count()
    
    def delete_vector_store(self) -> bool:
        """Delete vector store."""
        try:
            self.vector_store = None
            if Config.VECTOR_STORE_PATH.exists():
                shutil.rmtree(Config.VECTOR_STORE_PATH)
                Config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
            logger.info("Vector store deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")
            return False
    
    def get_loaded_files(self) -> List[str]:
        """Get list of loaded filenames from vector store metadata."""
        if not self.is_ready():
            return []
        
        try:
            # Get sample documents to extract unique filenames
            sample_docs = self.vector_store.similarity_search("", k=50)
            filenames = set()
            
            for doc in sample_docs:
                source = doc.metadata.get('source')
                if source and isinstance(source, str) and len(source) > 0:
                    # Basic validation - should have extension
                    if '.' in source and not source.startswith('.'):
                        filenames.add(source)
            
            result = sorted(list(filenames))
            
            if Config.DEBUG_MODE:
                logger.info(f"Found {len(result)} unique files in vector store")
                if Config.VERBOSE_MODE and result:
                    logger.info(f"Files: {result[:5]}{'...' if len(result) > 5 else ''}")
            
            return result if result else ["documents"]  # Fallback
            
        except Exception as e:
            logger.warning(f"Could not get loaded files: {e}")
            return ["documents"]  # Fallback
    
    def search_documents(self, query: str, k: int = None) -> List[Document]:
        """Direct document search for testing."""
        if not self.is_ready():
            logger.error("Vector store not ready for search")
            return []
        
        try:
            k = k or Config.RETRIEVAL_K
            retriever = self.get_retriever(search_kwargs={"k": k})
            if retriever:
                results = retriever.invoke(query)
                if Config.VERBOSE_MODE:
                    logger.info(f"Search '{query[:20]}...' returned {len(results)} results")
                return results
            return []
        except Exception as e:
            logger.error(f"Error in document search: {e}")
            return []