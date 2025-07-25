"""
Vector store management with hybrid embedding support.
Supports both OpenAI and Hugging Face embeddings.
"""

from typing import List, Optional, Any
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from config import config
from pydantic import SecretStr

class VectorStoreManager:
    """Manages vector store with flexible embedding providers."""
    
    def __init__(self):
        """Initialize with embedding model based on config."""
        if config.EMBEDDING_TYPE == "huggingface":
            print(f"Using Hugging Face embeddings: {config.HF_EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.HF_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
                encode_kwargs={'normalize_embeddings': True}
            )
        else:  # openai
            print(f"Using OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL}")
            
            self.embeddings = OpenAIEmbeddings(
                api_key=SecretStr(config.OPENAI_API_KEY),
                model=config.OPENAI_EMBEDDING_MODEL
            )
        
        self.vector_store: Optional[Chroma] = None
        self.documents: List[Document] = []
    
    def initialize_vector_store(self, documents: List[Document]) -> bool:
        """Initialize vector store with documents."""
        if not documents:
            print("ERROR: No documents provided")
            return False
        
        try:
            print("Initializing vector store...")
            config.ensure_directories()
            
            self.documents = documents
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(config.vector_store_path)
            )
            
            print(f"Vector store initialized with {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"ERROR initializing vector store: {e}")
            return False
    
    def load_vector_store(self) -> bool:
        """Load existing vector store from disk."""
        try:
            if not config.vector_store_path.exists() or not any(config.vector_store_path.iterdir()):
                print(f"Vector store not found at {config.vector_store_path}")
                return False
            
            self.vector_store = Chroma(
                persist_directory=str(config.vector_store_path),
                embedding_function=self.embeddings
            )
            
            print(f"Vector store loaded successfully")
            return True
            
        except Exception as e:
            print(f"ERROR loading vector store: {e}")
            return False
    
    def get_retriever(self) -> Optional[Any]:
        """Get retriever for document search."""
        if not self.vector_store:
            print("ERROR: Vector store not initialized")
            return None
        
        retriever = self.vector_store.as_retriever(
            search_type=config.SEARCH_TYPE,
            search_kwargs={"k": config.RETRIEVAL_K}
        )
        
        return retriever
    
    def similarity_search(self, query: str, k: int = 0) -> List[Document]:
        """Perform similarity search."""
        if not self.vector_store:
            return []
        
        k = k or config.RETRIEVAL_K
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self.vector_store is not None