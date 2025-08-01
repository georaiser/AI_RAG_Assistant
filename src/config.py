"""
Configurations with simplified validation and error handling.
"""

import os
import logging
from pathlib import Path
from typing import List, Literal, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

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
    #DOCUMENT_PATH: Path = Path("data/python-basics-sample-chapters.pdf")
    
    # Supported file formats
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.txt', '.md', '.rst', '.docx', '.dotx']
    
    # File size limits (in MB)
    MAX_FILE_SIZE_MB: int = 50
    WARN_FILE_SIZE_MB: int = 10
    
    # ===================
    # AI MODEL SETTINGS
    # ===================
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.4
    
    # Alternative models for different use cases
    AVAILABLE_MODELS: List[str] = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo-preview"
    ]
    
    # ===================
    # EMBEDDING SETTINGS
    # ===================
    EMBEDDING_TYPE: Literal["openai", "huggingface"] = "huggingface"
    
    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    AVAILABLE_OPENAI_EMBEDDINGS: List[str] = [
        "text-embedding-3-small",   # Fast & good
        "text-embedding-3-large",   # Better quality
        "text-embedding-ada-002"    # Legacy but reliable
    ]

    # HuggingFace Embeddings
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    AVAILABLE_HF_EMBEDDINGS: List[str] = [
        "sentence-transformers/all-MiniLM-L6-v2",   # Fast & good
        "sentence-transformers/all-mpnet-base-v2",  # Better quality
        "BAAI/bge-small-en-v1.5",                   # Excellent for retrieval
        "BAAI/bge-large-en-v1.5"                    # Best quality
    ]
    
    # ===================
    # TEXT PROCESSING SETTINGS
    # ===================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Improved separators for technical documents
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
    # RETRIEVAL SETTINGS
    # ===================
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "9"))
    SEARCH_TYPE: str = os.getenv("SEARCH_TYPE", "similarity")
    
    # Available search types
    AVAILABLE_SEARCH_TYPES: List[str] = [
        "similarity",
        "mmr", 
        "similarity_score_threshold"
    ]

    # MMR settings
    FETCH_K: int = int(os.getenv("FETCH_K", "25"))
    LAMBDA_MULT: float = float(os.getenv("LAMBDA_MULT", "0.7"))
    
    # Score threshold setting
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.3"))
        
    # ===================
    # APP SETTINGS
    # ===================
    APP_TITLE: str = "Docu Assistant"
    BOT_NAME: str = "Docu Assistant"
    DEBUG: bool = True
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None # "LOG_FILE.txt"  # Set to None for console only
    
    # Performance settings
    #MAX_TOKENS_PER_REQUEST: int = 4000
    
    # ===================
    # SIMPLIFIED VALIDATION
    # ===================
    
    @classmethod
    def validate(cls) -> bool:
        """Simplified validation - returns True if config is usable."""
        
        # Critical: API key must exist
        if not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is required")
            return False
        
        # Critical: Basic numeric validations
        if cls.CHUNK_SIZE <= 0:
            logger.error("CHUNK_SIZE must be positive")
            return False
            
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            logger.error("CHUNK_OVERLAP must be less than CHUNK_SIZE")
            return False
            
        if cls.RETRIEVAL_K <= 0:
            logger.error("RETRIEVAL_K must be positive")
            return False
        
        # Validate temperature range
        if not (0.0 <= cls.TEMPERATURE <= 2.0):
            logger.warning(f"TEMPERATURE {cls.TEMPERATURE} is outside recommended range [0.0, 2.0]")
        
        # Validate embedding type
        if cls.EMBEDDING_TYPE not in ["openai", "huggingface"]:
            logger.error(f"EMBEDDING_TYPE must be 'openai' or 'huggingface', got: {cls.EMBEDDING_TYPE}")
            return False
        
        # Validate search type
        if cls.SEARCH_TYPE not in cls.AVAILABLE_SEARCH_TYPES:
            logger.error(f"SEARCH_TYPE must be one of {cls.AVAILABLE_SEARCH_TYPES}, got: {cls.SEARCH_TYPE}")
            return False
        
        # Critical: Check if document exists (warn only)
        if not cls.DOCUMENT_PATH.exists():
            logger.warning(f"Document not found at {cls.DOCUMENT_PATH}")
        
        # Critical: Check file size
        if cls.DOCUMENT_PATH.exists():
            size_mb = cls.DOCUMENT_PATH.stat().st_size / (1024 * 1024)
            if size_mb > cls.MAX_FILE_SIZE_MB:
                logger.error(f"Document too large: {size_mb:.1f}MB > {cls.MAX_FILE_SIZE_MB}MB")
                return False
            elif size_mb > cls.WARN_FILE_SIZE_MB:
                logger.warning(f"Large document: {size_mb:.1f}MB > {cls.WARN_FILE_SIZE_MB}MB")
        
        logger.info("Configuration validation passed")
        return True
    
    @classmethod
    def setup_directories(cls) -> bool:
        """Create necessary directories."""
        try:
            cls.DATA_DIR.mkdir(exist_ok=True)
            cls.VECTOR_STORE_PATH.mkdir(exist_ok=True)
            logger.debug(f"Directories created: {cls.DATA_DIR}, {cls.VECTOR_STORE_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup basic logging."""
        try:
            log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        except (AttributeError, ValueError):
            log_level = logging.INFO
            logger.warning(f"Invalid LOG_LEVEL '{cls.LOG_LEVEL}', using INFO")
        
        # Configure logging format
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        
        if cls.LOG_FILE:
            try:
                logging.basicConfig(
                    level=log_level,
                    format=log_format,
                    filename=cls.LOG_FILE,
                    filemode='a',
                    force=True
                )
            except Exception as e:
                logger.error(f"Failed to setup file logging: {e}")
                # Fallback to console logging
                logging.basicConfig(level=log_level, format=log_format, force=True)
        else:
            logging.basicConfig(level=log_level, format=log_format, force=True)
        
        # Suppress noisy logs
        for logger_name in ["chromadb", "httpx", "openai", "langchain", "urllib3"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    @classmethod
    def get_summary(cls) -> dict:
        """Get basic configuration summary."""
        return {
            "model": cls.MODEL_NAME,
            "embedding_type": cls.EMBEDDING_TYPE,
            "chunk_size": cls.CHUNK_SIZE,
            "retrieval_k": cls.RETRIEVAL_K,
            "search_type": cls.SEARCH_TYPE,
            "document_exists": cls.DOCUMENT_PATH.exists(),
            "document_path": str(cls.DOCUMENT_PATH)
        }


# Create global config instance
config = Config()

# Simplified initialization
def initialize_config() -> bool:
    """Initialize configuration with minimal setup."""
    try:
        config.setup_logging()
        
        if not config.validate():
            return False
            
        if not config.setup_directories():
            return False
        
        if config.DEBUG:
            logger.info(f"Config: {config.get_summary()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        return False

# Initialize configuration
if not initialize_config():
    print("ERROR: Configuration failed. Check logs.")
    exit(1)