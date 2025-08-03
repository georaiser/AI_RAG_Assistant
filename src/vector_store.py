# vector_store.py
"""
Simplified vector store management with better error handling.
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
    """Simple vector store manager."""
    
    def __init__(self):
        """Initialize with embeddings."""
        self.embeddings = None
        self.vector_store = None
        self._create_embeddings()
    
    def _create_embeddings(self):
        """Create embedding model."""
        try:
            if Config.EMBEDDING_TYPE == "openai":
                self.embeddings = OpenAIEmbeddings(
                    model=Config.OPENAI_EMBEDDING_MODEL,
                    api_key=SecretStr(Config.OPENAI_API_KEY)
                )
            else:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=Config.HF_EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            logger.info(f"Embeddings created: {Config.EMBEDDING_TYPE}")
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    
    def initialize_vector_store(self, documents: List[Document]) -> bool:
        """Create new vector store - only when needed."""
        if not documents:
            logger.error("No documents provided")
            return False
        
        try:
            # Only create directory if it doesn't exist
            Config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
            
            # Filter valid documents
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            
            if not valid_docs:
                logger.error("No valid documents with content")
                return False
            
            logger.info(f"Creating vector store with {len(valid_docs)} documents")
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=valid_docs,
                embedding=self.embeddings,
                persist_directory=str(Config.VECTOR_STORE_PATH)
            )
            
            # Verify it worked
            count = self.vector_store._collection.count()
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
        """Load existing vector store."""
        try:
            # Check if directory exists and has content
            if not Config.VECTOR_STORE_PATH.exists():
                logger.info("Vector store directory doesn't exist")
                return False
            
            # Check for Chroma database files
            chroma_files = list(Config.VECTOR_STORE_PATH.glob("*.parquet")) + list(Config.VECTOR_STORE_PATH.glob("chroma.sqlite3"))
            if not chroma_files:
                logger.info("No Chroma database files found")
                return False
            
            # Load vector store
            self.vector_store = Chroma(
                persist_directory=str(Config.VECTOR_STORE_PATH),
                embedding_function=self.embeddings
            )
            
            # Verify it has documents
            count = self.vector_store._collection.count()
            if count == 0:
                logger.error("Vector store loaded but has no documents")
                self.vector_store = None
                return False
            
            logger.info(f"Vector store loaded with {count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.vector_store = None
            return False
    
    def is_ready(self) -> bool:
        """Check if vector store is ready."""
        if not self.vector_store:
            return False
        
        try:
            count = self.vector_store._collection.count()
            return count > 0
        except:
            return False
    
    def get_retriever(self):
        """Get retriever for document search."""
        if not self.is_ready():
            logger.error("Vector store not ready")
            return None
        
        try:
            search_kwargs = {"k": Config.RETRIEVAL_K}
            
            if Config.SEARCH_TYPE == "mmr":
                search_kwargs.update({
                    "fetch_k": Config.FETCH_K,
                    "lambda_mult": Config.LAMBDA_MULT
                })
            elif Config.SEARCH_TYPE == "similarity_score_threshold":
                search_kwargs["score_threshold"] = Config.SCORE_THRESHOLD
            
            retriever = self.vector_store.as_retriever(
                search_type=Config.SEARCH_TYPE,
                search_kwargs=search_kwargs
            )
            
            logger.info(f"Retriever created with {Config.SEARCH_TYPE}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            return None
    
    def get_document_count(self) -> int:
        """Get document count."""
        try:
            if self.vector_store:
                return self.vector_store._collection.count()
            return 0
        except:
            return 0
    
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