# vector_store.py
"""
Vector store management for RAG system.
Handles embedding creation and retrieval operations with enhanced logging.
"""

import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import Config
from pydantic import SecretStr

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations for document retrieval."""
    
    def __init__(self):
        self.embeddings = self._create_embeddings()
        self.vector_store = None
        logger.info(f"VectorStoreManager initialized with {Config.EMBEDDING_TYPE} embeddings")
    
    def _create_embeddings(self):
        """Create embedding model based on configuration."""
        try:
            if Config.EMBEDDING_TYPE == "openai":
                embeddings = OpenAIEmbeddings(
                    model=Config.OPENAI_EMBEDDING_MODEL,
                    api_key=SecretStr(Config.OPENAI_API_KEY)
                )
                logger.info(f"Created OpenAI embeddings: {Config.OPENAI_EMBEDDING_MODEL}")
            else:
                embeddings = HuggingFaceEmbeddings(
                    model_name=Config.HF_EMBEDDING_MODEL
                )
                logger.info(f"Created HuggingFace embeddings: {Config.HF_EMBEDDING_MODEL}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def initialize_vector_store(self, documents: List[Document]) -> bool:
        """Initialize vector store with documents."""
        if not documents:
            logger.error("No documents provided for vector store initialization")
            return False
        
        try:
            logger.info(f"Initializing vector store with {len(documents)} documents")
            
            # Ensure directory exists
            Config.VECTOR_STORE_PATH.mkdir(exist_ok=True)
            
            # Log document statistics
            text_docs = sum(1 for doc in documents if doc.metadata.get('content_type') != 'table')
            table_docs = sum(1 for doc in documents if doc.metadata.get('content_type') == 'table')
            logger.info(f"Document breakdown: {text_docs} text chunks, {table_docs} tables")
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(Config.VECTOR_STORE_PATH)
            )
            
            logger.info("Vector store initialized and persisted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False
    
    def load_vector_store(self) -> Optional[Chroma]:
        """Load existing vector store."""
        try:
            if not Config.VECTOR_STORE_PATH.exists():
                logger.warning("Vector store directory does not exist")
                return None
            
            # Check if directory has content
            if not any(Config.VECTOR_STORE_PATH.iterdir()):
                logger.warning("Vector store directory is empty")
                return None
            
            self.vector_store = Chroma(
                persist_directory=str(Config.VECTOR_STORE_PATH),
                embedding_function=self.embeddings
            )
            
            # Test the vector store
            collection = self.vector_store._collection
            if collection.count() == 0:
                logger.warning("Vector store is empty")
                return None
            
            logger.info(f"Vector store loaded successfully with {collection.count()} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
    
    def get_retriever(self):
        """Get retriever with configured search parameters."""
        if not self.vector_store:
            logger.error("Vector store not available for retriever creation")
            return None
        
        try:
            search_kwargs = {"k": Config.RETRIEVAL_K}
            
            if Config.SEARCH_TYPE == "mmr":
                search_kwargs.update({
                    "fetch_k": Config.FETCH_K,
                    "lambda_mult": Config.LAMBDA_MULT
                })
                logger.debug(f"Using MMR search with fetch_k={Config.FETCH_K}, lambda_mult={Config.LAMBDA_MULT}")
            elif Config.SEARCH_TYPE == "similarity_score_threshold":
                search_kwargs["score_threshold"] = Config.SCORE_THRESHOLD
                logger.debug(f"Using similarity threshold search with threshold={Config.SCORE_THRESHOLD}")
            else:
                logger.debug(f"Using similarity search with k={Config.RETRIEVAL_K}")
            
            retriever = self.vector_store.as_retriever(
                search_type=Config.SEARCH_TYPE,
                search_kwargs=search_kwargs
            )
            
            logger.info(f"Retriever created with search type: {Config.SEARCH_TYPE}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            return None
    
    def get_vector_store_stats(self) -> dict:
        """Get statistics about the vector store."""
        if not self.vector_store:
            return {"status": "not_available"}
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get sample metadata to understand document types
            if count > 0:
                sample_results = collection.peek(limit=min(10, count))
                metadata_list = sample_results.get('metadatas', [])
                
                sources = set()
                content_types = {}
                
                for metadata in metadata_list:
                    if metadata:
                        source = metadata.get('source', 'unknown')
                        sources.add(source)
                        
                        content_type = metadata.get('content_type', 'text')
                        content_types[content_type] = content_types.get(content_type, 0) + 1
                
                return {
                    "status": "ready",
                    "total_documents": count,
                    "unique_sources": len(sources),
                    "content_types": content_types,
                    "embedding_model": Config.OPENAI_EMBEDDING_MODEL if Config.EMBEDDING_TYPE == "openai" else Config.HF_EMBEDDING_MODEL
                }
            else:
                return {"status": "empty"}
                
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"status": "error", "error": str(e)}


# Convenience functions for backward compatibility
def initialize_vector_store(documents: List[Document]) -> bool:
    """Initialize vector store with documents."""
    manager = VectorStoreManager()
    return manager.initialize_vector_store(documents)


def load_vector_store() -> Optional[Chroma]:
    """Load existing vector store."""
    manager = VectorStoreManager()
    return manager.load_vector_store()