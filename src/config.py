"""
Configuration module for RAG applications.
Hybrid setup: Hugging Face embeddings + OpenAI LLM
"""

import os
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import field_validator

# Load environment variables
load_dotenv()

class Config(BaseSettings):
    """Configuration class with validation and type safety."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent
    
    # OpenAI configuration (for LLM only)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    
    # Embedding configuration
    EMBEDDING_TYPE: Literal["openai", "huggingface"] = "huggingface"  # Switch here!
    
    # OpenAI embedding (if using)
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Hugging Face embedding options
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast & good
    # Alternatives:
    # "sentence-transformers/all-mpnet-base-v2"  # Better quality
    # "BAAI/bge-small-en-v1.5"  # Excellent for retrieval
    
    # Vector store configuration
    VECTOR_STORE_DIR: str = "vector_store"
    
    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEPARATORS: List[str] = ["\n\n", "\n", " ", ""]
    
    # Retrieval configuration
    RETRIEVAL_K: int = 5
    FETCH_K: int = 10
    LAMBDA_MULT: float = 0.5
    SEARCH_TYPE: str = "similarity"
    
    # Document path
    DOCUMENT_PATH: str = "data/python-basics-sample-chapters.pdf"
    
    # App configuration
    APP_TITLE: str = "DocuPy Bot"
    APP_ICON: str = "🐍"
    BOT_NAME: str = "DocuPy Bot"
    BOT_DESCRIPTION: str = "Python documentation assistant"
    
    @field_validator('OPENAI_API_KEY')
    @classmethod
    def validate_api_key(cls, v):
        """Validate OpenAI API key is provided."""
        if not v or not v.strip():
            raise ValueError("OPENAI_API_KEY is required")
        return v
    
    @field_validator('TEMPERATURE')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @property
    def vector_store_path(self) -> Path:
        """Get absolute path to vector store directory."""
        return self.PROJECT_ROOT / self.VECTOR_STORE_DIR
    
    @property
    def document_full_path(self) -> Path:
        """Get absolute path to document."""
        doc_path = self.PROJECT_ROOT / self.DOCUMENT_PATH
        if doc_path.exists():
            return doc_path
        
        parent_path = self.PROJECT_ROOT.parent / self.DOCUMENT_PATH
        if parent_path.exists():
            return parent_path
            
        return doc_path
    
    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.vector_store_path.mkdir(exist_ok=True, parents=True)
    
# Global configuration instance
config = Config()