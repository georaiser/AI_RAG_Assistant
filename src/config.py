"""
Configuration module.
"""

import os
from pathlib import Path
from typing import List, Literal, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration with validation and smart defaults."""
    
    # ===================
    # PROJECT STRUCTURE
    # ===================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VECTOR_STORE_PATH: Path = PROJECT_ROOT / "vector_store"
    
    # ===================
    # DOCUMENT SETTINGS
    # ===================
    DOCUMENT_PATH: Path = Path("data/python-basics-sample-chapters.pdf")
    
    # Supported file formats
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.txt', '.md', '.rst', '.docx', '.dotx']
    
    # ===================
    # AI MODEL SETTINGS
    # ===================
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # ===================
    # EMBEDDING SETTINGS
    # ===================
    EMBEDDING_TYPE: Literal["openai", "huggingface"] = "huggingface"
    
    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # HuggingFace Embeddings (lightweight and fast)
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast & good
    # Alternatives:
    # "sentence-transformers/all-MiniLM-L6-v2"   # Fast & good
    # "sentence-transformers/all-mpnet-base-v2"  # Better quality
    # "BAAI/bge-small-en-v1.5"  # Excellent for retrieval
    
    # ===================
    # TEXT PROCESSING
    # ===================
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Text splitting separators (in order of preference)
    SEPARATORS: List[str] = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # ["\n\n", "\n", " ", ""]
    
    # ===================
    # RETRIEVAL SETTINGS
    # ===================
    RETRIEVAL_K: int = 7  # Number of final documents to return
    SEARCH_TYPE: str = "similarity"  # "similarity", "mmr", or "similarity_score_threshold"
    
    # MMR (Maximum Marginal Relevance) settings - only used when SEARCH_TYPE="mmr"
    FETCH_K: int = 20  # Docs to fetch before MMR filtering
    LAMBDA_MULT: float = 0.5  # Diversity vs relevance (0=max diversity, 1=max relevance)
    
    # Score threshold setting - only used when SEARCH_TYPE="similarity_score_threshold"
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.5"))  # Minimum similarity score (0-1)
    
    # ===================
    # APP SETTINGS
    # ===================
    APP_TITLE: str = "DocuPy Bot"
    APP_ICON: str = "🤖"
    BOT_NAME: str = "DocuPy Bot"
    BOT_DESCRIPTION: str = "Intelligent Document Assistant"
    
    # Streamlit settings
    PAGE_TITLE: str = "DocuPy Bot"
    PAGE_ICON: str = "🐍"
    LAYOUT: str = "wide"
    
    # ===================
    # DEVELOPMENT SETTINGS
    # ===================
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # ===================
    # VALIDATION METHODS
    # ===================
    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check required API key
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        # Check document path exists
        if not cls.DOCUMENT_PATH.exists():
            errors.append(f"Document path does not exist: {cls.DOCUMENT_PATH}")
        
        # Validate chunk settings
        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        # Validate retrieval settings
        if cls.RETRIEVAL_K <= 0:
            errors.append("RETRIEVAL_K must be positive")
        
        if cls.FETCH_K <= 0:
            errors.append("FETCH_K must be positive")
            
        if not 0 <= cls.LAMBDA_MULT <= 1:
            errors.append("LAMBDA_MULT must be between 0 and 1")
        
        if cls.SEARCH_TYPE not in ["similarity", "mmr", "similarity_score_threshold"]:
            errors.append("SEARCH_TYPE must be 'similarity', 'mmr', or 'similarity_score_threshold'")
        
        if not 0 <= cls.SCORE_THRESHOLD <= 1:
            errors.append("SCORE_THRESHOLD must be between 0 and 1")
        
        # Validate embedding type
        if cls.EMBEDDING_TYPE not in ["openai", "huggingface"]:
            errors.append("EMBEDDING_TYPE must be 'openai' or 'huggingface'")
        
        return errors
    
    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_PATH.mkdir(exist_ok=True)
    
    @classmethod
    def get_document_files(cls) -> List[Path]:
        """Get all supported document files from document path."""
        files = []
        
        if cls.DOCUMENT_PATH.is_file():
            # Single file
            if cls.DOCUMENT_PATH.suffix.lower() in cls.SUPPORTED_FORMATS:
                files.append(cls.DOCUMENT_PATH)
        elif cls.DOCUMENT_PATH.is_dir():
            # Directory - find all supported files
            for ext in cls.SUPPORTED_FORMATS:
                files.extend(cls.DOCUMENT_PATH.glob(f"*{ext}"))
        
        return sorted(files)
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (for debugging)."""
        print("=" * 50)
        print("DOCUPY BOT CONFIGURATION")
        print("=" * 50)
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Embedding: {cls.EMBEDDING_TYPE} ({cls.OPENAI_EMBEDDING_MODEL if cls.EMBEDDING_TYPE == 'openai' else cls.HF_EMBEDDING_MODEL})")
        print(f"Document Path: {cls.DOCUMENT_PATH}")
        print(f"Documents Found: {len(cls.get_document_files())}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Retrieval K: {cls.RETRIEVAL_K} (fetch_k: {cls.FETCH_K}, λ: {cls.LAMBDA_MULT})")
        print(f"Search Type: {cls.SEARCH_TYPE}")
        print(f"Vector Store: {cls.VECTOR_STORE_PATH}")
        print("=" * 50)
    
    @classmethod
    def get_model_config(cls) -> dict:
        """Get model configuration for easy passing."""
        return {
            "model": cls.MODEL_NAME,
            "temperature": cls.TEMPERATURE,
            "api_key": cls.OPENAI_API_KEY
        }
    
    @classmethod
    def get_embedding_config(cls) -> dict:
        """Get embedding configuration."""
        if cls.EMBEDDING_TYPE == "openai":
            return {
                "type": "openai",
                "model": cls.OPENAI_EMBEDDING_MODEL,
                "api_key": cls.OPENAI_API_KEY
            }
        else:
            return {
                "type": "huggingface",
                "model": cls.HF_EMBEDDING_MODEL
            }
    
    @classmethod
    def get_retrieval_config(cls) -> dict:
        """Get retrieval configuration."""
        return {
            "k": cls.RETRIEVAL_K,
            "search_type": cls.SEARCH_TYPE,
            "fetch_k": cls.FETCH_K,
            "lambda_mult": cls.LAMBDA_MULT
        }
        """Get text chunking configuration."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "separators": cls.SEPARATORS
        }


# Global configuration instance
config = Config()

# Auto-validate on import (optional - remove if you don't want this)
if __name__ != "__main__":
    validation_errors = config.validate()
    if validation_errors and config.DEBUG:
        print("⚠️  Configuration Issues:")
        for error in validation_errors:
            print(f"  - {error}")
    
    # Setup directories
    config.setup_directories()

# For development/debugging
if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    
    # Print config
    config.print_config()
    
    # Validate
    errors = config.validate()
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ Configuration is valid!")
    
    # Test document discovery
    docs = config.get_document_files()
    print(f"\n📄 Found {len(docs)} document(s):")
    for doc in docs:
        size_kb = doc.stat().st_size / 1024 if doc.exists() else 0
        print(f"  - {doc.name} ({size_kb:.1f} KB)")