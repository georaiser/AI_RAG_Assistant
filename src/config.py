"""
Configurations.
"""

import os
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:

    # ===================
    # PROJECT STRUCTURE
    # ===================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VECTOR_STORE_PATH: Path = PROJECT_ROOT / "vector_store"
    
    # ===================
    # DOCUMENT SETTINGS
    # ===================
    DOCUMENT_PATH: Path = PROJECT_ROOT / "data" / "python-basics-sample-chapters.pdf"
    
    # Supported file formats
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.txt', '.md', '.rst', '.docx', '.dotx']
    
    # ===================
    # AI MODEL SETTINGS
    # ===================
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.3  # Lower for more consistent answers
    
    # ===================
    # EMBEDDING SETTINGS
    # ===================
    EMBEDDING_TYPE: Literal["openai", "huggingface"] = "huggingface"
    
    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    # Alternatives:
    # "text-embedding-3-small"  # Fast & good
    # "text-embedding-3-large"  # Better quality

    # HuggingFace Embeddings
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternatives:
    # "sentence-transformers/all-MiniLM-L6-v2"   # Fast & good
    # "sentence-transformers/all-mpnet-base-v2"  # Better quality
    # "BAAI/bge-small-en-v1.5"  # Excellent for retrieval
    # "BAAI/bge-large-en-v1.5"  # Best quality
            
    # ===================
    # IMPROVED TEXT PROCESSING
    # ===================
    CHUNK_SIZE: int = 1500  # Increased for better context
    CHUNK_OVERLAP: int = 300  # Increased overlap for better continuity
    # Better separators for technical documents
    SEPARATORS: List[str] = [
        "\n\n\n",  # Multiple line breaks
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentence endings
        "! ",      # Exclamation endings
        "? ",      # Question endings
        "; ",      # Semicolon breaks
        ", ",      # Comma breaks (for lists)
        " ",       # Word breaks
        ""         # Character breaks (last resort)
    ]
    
    # ===================
    # ENHANCED RETRIEVAL SETTINGS
    # ===================
    RETRIEVAL_K: int = 12  # Increased for better coverage
    SEARCH_TYPE: str = "similarity"  # More reliable than MMR for most cases
    # Alternatives:
    # "similarity"  # Use similarity search for most cases
    # "mmr"  # Use MMR for diversity
    # "similarity_score_threshold"  # Use score threshold for filtering

    # MMR settings (if using MMR)
    FETCH_K: int = 25  # Increased for better selection
    LAMBDA_MULT: float = 0.7  # Higher diversity
    
    # Score threshold setting (more permissive)
    SCORE_THRESHOLD: float = 0.3  # Lower threshold for more results
    
    # Fallback retrieval settings
    ENABLE_FALLBACK_SEARCH: bool = True
    FALLBACK_K: int = 15  # Even more documents for fallback
    
    # Query expansion settings
    ENABLE_QUERY_EXPANSION: bool = True
    TEMPERATURE_EXPANSION: float = 0.4  # Slightly higher for creativity
    MAX_QUERY_VARIATIONS: int = 3  # Multiple query variations
        
    # ===================
    # APP SETTINGS
    # ===================
    APP_TITLE: str = "Docu Bot"
    BOT_NAME: str = "Docu Bot"
    DEBUG: bool = True  # Enable debug mode for detailed logs
    
    # Response improvement settings
    ENABLE_CONTEXT_ENHANCEMENT: bool = True
    MIN_CONTEXT_LENGTH: int = 100  # Minimum context to attempt answer
    
    # ===================
    # ESSENTIAL VALIDATION
    # ===================
    @classmethod
    def validate_essentials(cls) -> bool:
        """Validate only essential requirements."""
        if not cls.OPENAI_API_KEY:
            print("Error: OPENAI_API_KEY is required")
            return False
        return True
    
    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_PATH.mkdir(exist_ok=True)
    
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
        """Get optimized retrieval configuration."""
        base_config = {
            "k": cls.RETRIEVAL_K,
            "search_type": cls.SEARCH_TYPE
        }
        
        if cls.SEARCH_TYPE == "mmr":
            base_config.update({
                "fetch_k": cls.FETCH_K,
                "lambda_mult": cls.LAMBDA_MULT
            })
        elif cls.SEARCH_TYPE == "similarity_score_threshold":
            base_config.update({
                "score_threshold": cls.SCORE_THRESHOLD
            })
        
        return base_config


# Global configuration instance
config = Config()

# Essential validation only
if __name__ != "__main__":
    if not config.validate_essentials():
        exit(1)
    config.setup_directories()

# For development/debugging
if __name__ == "__main__":
    print("=" * 60)
    print("DOCUPY BOT - IMPROVED CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Temperature: {config.TEMPERATURE}")
    print(f"Embedding: {config.EMBEDDING_TYPE}")
    print(f"Document Path: {config.DOCUMENT_PATH}")
    print(f"Chunk Size: {config.CHUNK_SIZE}")
    print(f"Chunk Overlap: {config.CHUNK_OVERLAP}")
    print(f"Retrieval K: {config.RETRIEVAL_K}")
    print(f"Search Type: {config.SEARCH_TYPE}")
    print(f"Query Expansion: {config.ENABLE_QUERY_EXPANSION}")
    print(f"Fallback Search: {config.ENABLE_FALLBACK_SEARCH}")
    print(f"Debug Mode: {config.DEBUG}")
    print("=" * 60)