"""
Simplified configuration module with essential settings only.
"""

import os
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Simplified configuration with essential settings."""
    
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
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    
    # ===================
    # EMBEDDING SETTINGS
    # ===================
    EMBEDDING_TYPE: Literal["openai", "huggingface"] = "openai"
    
    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # HuggingFace Embeddings
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternatives:
    # "sentence-transformers/all-MiniLM-L6-v2"   # Fast & good
    # "sentence-transformers/all-mpnet-base-v2"  # Better quality
    # "BAAI/bge-small-en-v1.5"  # Excellent for retrieval

        
    # ===================
    # TEXT PROCESSING
    # ===================
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEPARATORS: List[str] = ["\n\n", "\n", ". ", "! ", "? ", " ", ""] # ["\n\n", "\n", " ", ""]
    
    # ===================
    # RETRIEVAL SETTINGS
    # ===================
    RETRIEVAL_K: int = 7
    SEARCH_TYPE: str = "mmr"  # "similarity", "mmr", or "similarity_score_threshold"
    
    # MMR settings
    FETCH_K: int = 20
    LAMBDA_MULT: float = 0.5
    
    # Score threshold setting
    SCORE_THRESHOLD: float = 0.5

    # Query expansion (enable/disable)
    ENABLE_QUERY_EXPANSION: bool = True
    TEMPERATURE_EXPANSION: float = 0.5
        
    # ===================
    # APP SETTINGS
    # ===================
    APP_TITLE: str = "Docu Bot"
    BOT_NAME: str = "Docu Bot"
    DEBUG: bool = True
    
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


# Global configuration instance
config = Config()

# Essential validation only
if __name__ != "__main__":
    if not config.validate_essentials():
        exit(1)
    config.setup_directories()

# For development/debugging
if __name__ == "__main__":
    print("=" * 50)
    print("DOCUPY BOT CONFIGURATION")
    print("=" * 50)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Embedding: {config.EMBEDDING_TYPE}")
    print(f"Document Path: {config.DOCUMENT_PATH}")
    print(f"Query Expansion: {config.ENABLE_QUERY_EXPANSION}")
    print(f"Search Type: {config.SEARCH_TYPE}")
    print(f"Retrieval K: {config.RETRIEVAL_K}")
    print("=" * 50)