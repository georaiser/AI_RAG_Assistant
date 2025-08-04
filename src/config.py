# config.py
"""
Configuration settings for the RAG project with improved validation.
"""

import os
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Project structure
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VECTOR_STORE_PATH: Path = PROJECT_ROOT / "vector_store"

    # Document settings
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.txt', '.md', '.rst', '.docx', '.dotx']
    
    # File processing mode
    PROCESS_MULTIPLE_DOCUMENTS: bool = True
    
    # Single file path (when PROCESS_MULTIPLE_DOCUMENTS = False)
    SINGLE_DOCUMENT_PATH: Path = DATA_DIR / "python-basics-sample-chapters.pdf"
    
    # Multiple documents directory (when PROCESS_MULTIPLE_DOCUMENTS = True)
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"

    # AI model settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.2
    
    # Embedding settings
    EMBEDDING_TYPE: Literal["openai", "huggingface"] = "huggingface"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    AVAILABLE_OPENAI_EMBEDDINGS: List[str] = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ]
    
    AVAILABLE_HF_EMBEDDINGS: List[str] = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5"
    ]

    # Text processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval settings
    MAX_CONTEXT_CHARS: int = 12000
    SUMMARY_K: int = 20
    RETRIEVAL_K: int = 6
    SEARCH_TYPE: str = "mmr"

    AVAILABLE_SEARCH_TYPES: List[str] = [
        "similarity",
        "mmr",
        "similarity_score_threshold"
    ]    

    # MMR parameters
    FETCH_K: int = 12
    LAMBDA_MULT: float = 0.7
    SCORE_THRESHOLD: float = 0.4
    
    # App settings
    APP_TITLE: str = "Technical Document Assistant"
    
    # Debug settings
    DEBUG_MODE: bool = True  
    VERBOSE_MODE: bool = False  # Changed to False by default
        
    @classmethod
    def validate(cls) -> bool:
        """Validate critical settings with better error messages."""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")
            
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if cls.RETRIEVAL_K <= 0:
            errors.append("RETRIEVAL_K must be positive")
            
        if cls.FETCH_K < cls.RETRIEVAL_K:
            errors.append("FETCH_K must be >= RETRIEVAL_K")
        
        if not 0.0 <= cls.LAMBDA_MULT <= 1.0:
            errors.append("LAMBDA_MULT must be between 0.0 and 1.0")

        if cls.EMBEDDING_TYPE not in ["openai", "huggingface"]:
            errors.append("EMBEDDING_TYPE must be 'openai' or 'huggingface'")
        
        if cls.SEARCH_TYPE not in cls.AVAILABLE_SEARCH_TYPES:
            errors.append(f"SEARCH_TYPE must be one of {cls.AVAILABLE_SEARCH_TYPES}")
        
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return False
        
        return True
    
    @classmethod
    def setup_directories(cls) -> bool:
        """Create necessary directories with better error handling."""
        try:
            cls.DATA_DIR.mkdir(exist_ok=True)
            cls.VECTOR_STORE_PATH.mkdir(exist_ok=True)
            
            if cls.PROCESS_MULTIPLE_DOCUMENTS:
                cls.DOCUMENTS_DIR.mkdir(exist_ok=True)
                print(f"Created documents directory: {cls.DOCUMENTS_DIR}")
            else:
                if not cls.SINGLE_DOCUMENT_PATH.exists():
                    print(f"WARNING: Single document not found: {cls.SINGLE_DOCUMENT_PATH}")
            
            return True
        except Exception as e:
            print(f"ERROR: Failed to create directories: {e}")
            return False