# config.py
"""
Configuration settings for the RAG project.
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
    MAX_FILE_SIZE_MB: int = 50
    
    # File processing mode
    PROCESS_MULTIPLE_DOCUMENTS: bool = True
    
    # Single file path (when PROCESS_MULTIPLE_DOCUMENTS = False)
    SINGLE_DOCUMENT_PATH: Path = DATA_DIR / "python-basics-sample-chapters.pdf"
    
    # Multiple documents directory (when PROCESS_MULTIPLE_DOCUMENTS = True)
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"

    # AI model settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.5
    
    # Embedding settings
    EMBEDDING_TYPE: Literal["openai", "huggingface"] = "openai"
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
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200
    
    # Retrieval settings
    RETRIEVAL_K: int = 6
    SEARCH_TYPE: str = "similarity"
    
    AVAILABLE_SEARCH_TYPES: List[str] = [
        "similarity",
        "mmr",
        "similarity_score_threshold"
    ]
    
    # Search parameters
    FETCH_K: int = 20
    LAMBDA_MULT: float = 0.7
    SCORE_THRESHOLD: float = 0.4
    
    # App settings
    APP_TITLE: str = "RAG Document Assistant"
        
    @classmethod
    def validate(cls) -> bool:
        """Validate critical settings."""
        if not cls.OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY is required")
            return False
        
        if cls.CHUNK_SIZE <= 0 or cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            print("ERROR: Invalid chunk settings")
            return False
        
        return True
    
    @classmethod
    def setup_directories(cls) -> bool:
        """Create necessary directories."""
        try:
            cls.DATA_DIR.mkdir(exist_ok=True)
            cls.VECTOR_STORE_PATH.mkdir(exist_ok=True)
            if cls.PROCESS_MULTIPLE_DOCUMENTS:
                cls.DOCUMENTS_DIR.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"ERROR: Failed to create directories: {e}")
            return False