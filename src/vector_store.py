"""
Vector store with hybrid search capabilities.
Combines semantic search with optional keyword-based retrieval.
"""

import logging
import pickle
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from src.config import config

logger = logging.getLogger(__name__)


class SemanticSearchManager:
    """Manages semantic search functionality."""
    
    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self.embeddings = self._initialize_embeddings()
        self._vector_store: Optional[Chroma] = None
    
    def _initialize_embeddings(self):
        """Initialize embedding model based on config."""
        if config.EMBEDDING_TYPE == "huggingface":
            logger.info(f"Initializing HuggingFace embeddings: {config.HF_EMBEDDING_MODEL}")
            return HuggingFaceEmbeddings(
                model_name=config.HF_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},  # Changed from 'cuda' to 'cpu' for broader compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            logger.info(f"Initializing OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL}")
            return OpenAIEmbeddings(
                api_key=config.OPENAI_API_KEY,  
                model=config.OPENAI_EMBEDDING_MODEL
            )
    
    def initialize(self, documents: List[Document]) -> bool:
        """Initialize vector store with documents."""
        if not documents:
            logger.error("No documents provided for semantic search initialization")
            return False
        
        # FIX: Validate that all items are Document objects
        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                logger.error(f"Item {i} is not a Document object: {type(doc)}")
                return False
            
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating vector store with {len(documents)} documents")
            self._vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            logger.info(f"Semantic search initialized with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Semantic search initialization failed: {e}")
            self._vector_store = None
            return False
    
    def load_from_disk(self) -> bool:
        """Load existing vector store from disk."""
        try:
            if not self._exists():
                logger.warning("Vector store does not exist on disk")
                return False
            
            self._vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            
            logger.info("Semantic search loaded from disk")
            return True
            
        except Exception as e:
            logger.error(f"Semantic search load failed: {e}")
            self._vector_store = None
            return False
    
    def _exists(self) -> bool:
        """Check if vector store exists on disk."""
        return (
            self.persist_directory.exists() and 
            self.persist_directory.is_dir() and 
            any(self.persist_directory.iterdir())
        )
    
    def get_retriever(self, k: int):
        """Get semantic retriever with specified k value."""
        if not self._vector_store:
            logger.error("Vector store not initialized")
            return None
        return self._vector_store.as_retriever(search_kwargs={"k": k})
    
    @property
    def is_available(self) -> bool:
        """Check if semantic search is available."""
        return self._vector_store is not None
    
    def get_vector_store(self) -> Optional[Chroma]:
        """Get the underlying vector store."""
        return self._vector_store


class BM25Manager:
    """Manages BM25 retriever functionality."""
    
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._is_enabled = getattr(config, 'BM25_RETRIEVER', False)
        
        if self._is_enabled:
            logger.info("BM25 retriever enabled")
        else:
            logger.info("BM25 retriever disabled")
    
    @property
    def is_enabled(self) -> bool:
        return self._is_enabled
    
    @property  
    def is_available(self) -> bool:
        return self._is_enabled and self._bm25_retriever is not None
    
    def initialize(self, documents: List[Document]) -> bool:
        """Initialize BM25 retriever with documents."""
        if not self._is_enabled:
            logger.info("BM25 retriever disabled, skipping initialization")
            return False
            
        if not documents:
            logger.error("No documents provided for BM25 initialization")
            return False
        
        # FIX: Validate that all items are Document objects
        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                logger.error(f"BM25 initialization: Item {i} is not a Document object: {type(doc)}")
                return False
            
        try:
            logger.info(f"Initializing BM25 retriever with {len(documents)} documents")
            self._bm25_retriever = BM25Retriever.from_documents(documents)
            self._bm25_retriever.k = max(1, config.RETRIEVAL_K // 2)  # FIX: Ensure k is at least 1
            self._cache_to_disk()
            logger.info("BM25 retriever initialized successfully")
            return True
        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}")
            self._bm25_retriever = None
            return False
    
    def load_from_cache(self) -> bool:
        """Load BM25 retriever from cache."""
        if not self._is_enabled:
            logger.info("BM25 retriever disabled, skipping cache load")
            return False
            
        if not self.cache_path.exists():
            logger.info("BM25 cache file does not exist")
            return False
            
        try:
            with open(self.cache_path, 'rb') as f:
                self._bm25_retriever = pickle.load(f)
            logger.info("BM25 retriever loaded from cache")
            return True
        except Exception as e:
            logger.error(f"BM25 cache load failed: {e}")
            self._cleanup_cache()
            return False
    
    def _cache_to_disk(self) -> None:
        """Cache BM25 retriever to disk."""
        if not self._bm25_retriever:
            return
            
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self._bm25_retriever, f)
            logger.info("BM25 retriever cached to disk")
        except Exception as e:
            logger.error(f"BM25 cache save failed: {e}")
    
    def _cleanup_cache(self) -> None:
        """Clean up corrupted cache file."""
        try:
            if self.cache_path.exists():
                self.cache_path.unlink()
                logger.info("Cleaned up corrupted BM25 cache")
        except Exception as e:
            logger.error(f"Failed to cleanup BM25 cache: {e}")
    
    def get_retriever(self, k: int) -> Optional[BM25Retriever]:
        """Get BM25 retriever with specified k value."""
        if not self.is_available:
            return None
            
        if self._bm25_retriever is None:
            logger.error("BM25 retriever is None")
            return None
            
        # FIX: Ensure k is at least 1
        self._bm25_retriever.k = max(1, k // 2)
        return self._bm25_retriever
    
    def rebuild(self, documents: List[Document]) -> bool:
        """Rebuild BM25 retriever with new documents."""
        return self.initialize(documents)


class VectorStoreManager:
    """Vector store with hybrid search capabilities."""
    
    def __init__(self):
        self.persist_directory = config.VECTOR_STORE_PATH
        
        # Initialize search managers
        self.semantic_manager = SemanticSearchManager(self.persist_directory)
        self.bm25_manager = BM25Manager(self.persist_directory / "bm25_cache.pkl")
        
        # Retriever cache
        self._retriever_cache: Dict[int, Any] = {}
        
        logger.info("Vector store manager initialized")
    
    def initialize_vector_store(self, documents: List[Document]) -> Optional[Chroma]:
        """Initialize both semantic and BM25 search with documents."""
        if not documents:
            logger.error("No documents provided for vector store initialization")
            return None
        
        # FIX: Additional validation
        logger.info(f"Validating {len(documents)} documents for vector store initialization")
        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                logger.error(f"Document {i} is not a Document object: {type(doc)}")
                return None
        
        # Initialize semantic search 
        if not self.semantic_manager.initialize(documents):
            logger.error("Semantic search initialization failed")
            return None
        
        # Initialize BM25 search (optional)
        bm25_success = self.bm25_manager.initialize(documents)
        
        # Clear cache
        self._retriever_cache.clear()
        
        # Log initialization results
        if bm25_success:
            logger.info("Initialized with hybrid search (semantic + BM25)")
        else:
            logger.info("Initialized with semantic search only")

        logger.info(f"Vector store initialized successfully with {len(documents)} documents")
        return self.semantic_manager.get_vector_store()
    
    def load_vector_store(self) -> Optional[Chroma]:
        """Load existing search systems from disk."""
        # Load semantic search
        semantic_loaded = self.semantic_manager.load_from_disk()
        if not semantic_loaded:
            logger.warning("Semantic search could not be loaded")
            return None
        
        # Load BM25 search (optional)
        bm25_loaded = self.bm25_manager.load_from_cache()
        
        # Clear cache when loading
        self._retriever_cache.clear()
        
        # Log load results
        if bm25_loaded:
            logger.info("Loaded with hybrid search (semantic + BM25)")
        else:
            logger.info("Loaded with semantic search only")
        
        return self.semantic_manager.get_vector_store()
    
    def get_retriever(self, k: int):
        """Get retriever with optional hybrid search and caching."""
        # FIX: Ensure k is at least 1
        k = max(1, k)
        
        # Check cache
        if k in self._retriever_cache:
            logger.debug(f"Using cached retriever for k={k}")
            return self._retriever_cache[k]
        
        # Ensure semantic search is loaded
        if not self.semantic_manager.is_available:
            logger.info("Semantic search not available, attempting to load from disk")
            if not self.load_vector_store():
                raise ValueError("Could not load or initialize semantic search")
        
        # Get semantic retriever
        semantic_retriever = self.semantic_manager.get_retriever(k)
        if not semantic_retriever:
            raise ValueError("Could not create semantic retriever")
        
        # Create hybrid retriever if BM25 is available
        bm25_retriever = self.bm25_manager.get_retriever(k)
        if bm25_retriever:
            try:
                retriever = EnsembleRetriever(
                    retrievers=[semantic_retriever, bm25_retriever],
                    weights=[0.7, 0.3]
                )
                logger.debug(f"Using hybrid retriever with k={k}")
            except Exception as e:
                logger.warning(f"Failed to create ensemble retriever: {e}, falling back to semantic only")
                retriever = semantic_retriever
        else:
            retriever = semantic_retriever
            logger.debug(f"Using semantic retriever with k={k}")
        
        # Cache and return
        self._retriever_cache[k] = retriever
        return retriever
    
    def is_semantic_available(self) -> bool:
        """Check if semantic search is available."""
        return self.semantic_manager.is_available
    
    def is_bm25_available(self) -> bool:
        """Check if BM25 retriever is available."""
        return self.bm25_manager.is_available
    
    def is_bm25_enabled(self) -> bool:
        """Check if BM25 retriever is enabled in config."""
        return self.bm25_manager.is_enabled
    
    def rebuild_bm25(self, documents: List[Document]) -> bool:
        """Rebuild BM25 retriever with new documents."""
        success = self.bm25_manager.rebuild(documents)
        if success:
            self._retriever_cache.clear()
        return success
    
    def clear_cache(self) -> None:
        """Clear retriever cache."""
        self._retriever_cache.clear()
        logger.info("Retriever cache cleared")


# Global vector store manager
_vector_store_manager = VectorStoreManager()

def initialize_vector_store(documents: List[Document]) -> Optional[Chroma]:
    """Initialize vector store with documents."""
    return _vector_store_manager.initialize_vector_store(documents)

def get_retriever(k: int):
    """Get retriever with optional hybrid search."""
    return _vector_store_manager.get_retriever(k)

def load_vector_store() -> Optional[Chroma]:
    """Load existing vector store."""
    return _vector_store_manager.load_vector_store()

def is_semantic_available() -> bool:
    """Check if semantic search is available."""
    return _vector_store_manager.is_semantic_available()

def is_bm25_available() -> bool:
    """Check if BM25 retriever is available."""
    return _vector_store_manager.is_bm25_available()

def is_bm25_enabled() -> bool:
    """Check if BM25 retriever is enabled in config."""
    return _vector_store_manager.is_bm25_enabled()

def rebuild_bm25(documents: List[Document]) -> bool:
    """Rebuild BM25 retriever with new documents."""
    return _vector_store_manager.rebuild_bm25(documents)