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
    DOCUMENT_PATH: Path = Path("data/cleaned.pdf")
    
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
        
    # ===================
    # APP SETTINGS
    # ===================
    APP_TITLE: str = "Docu Bot"
    BOT_NAME: str = "Docu Bot"
    DEBUG: bool = True  # Enable debug mode for detailed logs
    
    # ===================
    # ESSENTIAL VALIDATION
    # ===================
    
    @classmethod
    def validate(cls) -> bool:
        """Validate essential requirements."""
        return bool(cls.OPENAI_API_KEY)
    
    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_PATH.mkdir(exist_ok=True)

config = Config()

if not config.validate():
    print("Error: OPENAI_API_KEY is required")
    exit(1)

config.setup_directories()