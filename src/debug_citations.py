# debug_citations.py
"""
Debug script to verify citation data flow and page tracking.
Run this to check if your documents have proper page metadata.
"""

import logging
from pathlib import Path
from data_loader import load_data
from vector_store import VectorStoreManager
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_document_metadata():
    """Debug document loading and metadata."""
    print("=== DEBUGGING DOCUMENT METADATA ===\n")
    
    # Load documents
    documents, file_info = load_data()
    print(f"Loaded {len(documents)} documents")
    print(f"File info: {file_info}\n")
    
    # Check metadata for first few documents
    print("=== SAMPLE DOCUMENT METADATA ===")
    for i, doc in enumerate(documents[:10]):
        metadata = doc.metadata
        content_preview = doc.page_content[:100].replace('\n', ' ')
        
        print(f"Document {i}:")
        print(f"  Source: {metadata.get('source', 'MISSING')}")
        print(f"  Page: {metadata.get('page', 'MISSING')}")
        print(f"  Content Type: {metadata.get('content_type', 'MISSING')}")
        print(f"  File Type: {metadata.get('file_type', 'MISSING')}")
        print(f"  Content Preview: {content_preview}...")
        print()
    
    return documents

def debug_vector_store():
    """Debug vector store and retrieval."""
    print("=== DEBUGGING VECTOR STORE ===\n")
    
    vector_manager = VectorStoreManager()
    vector_store = vector_manager.load_vector_store()
    
    if not vector_store:
        print("Vector store not found. Creating new one...")
        documents, _ = load_data()
        if vector_manager.initialize_vector_store(documents):
            print("Vector store created successfully")
            vector_store = vector_manager.load_vector_store()
        else:
            print("Failed to create vector store")
            return
    
    # Test retrieval
    retriever = vector_manager.get_retriever()
    if retriever:
        print("Testing retrieval with sample query...")
        docs = retriever.get_relevant_documents("technical documentation methods")
        
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs[:5]):
            metadata = doc.metadata
            print(f"  Doc {i}: {metadata.get('source', 'NO_SOURCE')} - Page: {metadata.get('page', 'NO_PAGE')}")
            print(f"    Content: {doc.page_content[:80]}...")
            print()

def debug_enhanced_context():
    """Debug enhanced context creation."""
    print("=== DEBUGGING ENHANCED CONTEXT ===\n")
    
    from backend import RAGEngine
    
    engine = RAGEngine()
    if not engine.vector_store:
        print("Vector store not ready")
        return
    
    # Get sample documents
    retriever = engine.vector_manager.get_retriever()
    docs = retriever.get_relevant_documents("technical methods")
    
    # Create enhanced context
    enhanced_context = engine._create_enhanced_context(docs)
    
    print("Enhanced Context Preview:")
    print("=" * 50)
    print(enhanced_context[:1000] + "..." if len(enhanced_context) > 1000 else enhanced_context)
    print("=" * 50)

if __name__ == "__main__":
    print("Starting citation debugging...\n")
    
    # Debug steps
    debug_document_metadata()
    debug_vector_store()
    debug_enhanced_context()
    
    print("Debugging complete!")
