"""
Configuration module for the Bot application.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Optional, List, Literal
from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class that manages all application settings."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent

    # Document path
    DOCUMENT_PATH: str = "data/python-basics-sample-chapters.pdf"

    # OpenAI configuration
    OPENAI_API_KEY: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7

    # Embedding configuration
    EMBEDDING_TYPE: Literal["openai", "huggingface"] = "openai"

    # OpenAI embedding
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Hugging Face embedding options
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast & good
    # Alternatives:
    # "sentence-transformers/all-mpnet-base-v2"  # Better quality
    # "BAAI/bge-small-en-v1.5"  # Excellent for retrieval

    # Vector store configuration
    VECTOR_STORE_PATH: Path = PROJECT_ROOT /  "vector_store"

    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEPARATORS: List[str] = ["\n\n", "\n", " ", ""]  # ["\n\n\n", "\n\n", "\n", ".", "!", "?", " ", ""],

    # Retrieval configuration
    RETRIEVAL_K: int = 10
    FETCH_K: int = 40
    LAMBDA_MULT: float = 0.5
    SEARCH_TYPE: str = "similarity"

    # BM25 Retriever Configuration
    BM25_RETRIEVER = True  # Set to False to disable BM25 hybrid search
    # Alternative configurations you might want to add:
    # BM25_WEIGHT = 0.3  # Weight for BM25 in ensemble (semantic gets 1 - BM25_WEIGHT)
    # BM25_K_RATIO = 0.5  # Ratio of total k to use for BM25 (default: k // 2)

    # App configuration
    APP_TITLE: str = "Docu Bot"
    APP_ICON: str = "🐍"
    BOT_NAME: str = "Docu Bot"
    BOT_DESCRIPTION: str = "Documentation assistant"    

# Global configuration instance
config = Config()