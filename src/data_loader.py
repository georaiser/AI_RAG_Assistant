"""
Document processor for extracting and chunking text from various document formats.
Supports PDF files with extensible architecture for other formats.
"""

import pdfplumber
from pathlib import Path
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config import config

class PDFProcessor:
    """PDF document processor using pdfplumber."""
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            full_text = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add non-empty pages
                        full_text.append(page_text)
            
            extracted_text = "\n\n".join(full_text)
            print(f"Total characters extracted: {len(extracted_text)}")
            return extracted_text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

class DocumentLoader:
    """Main document loader class that handles text extraction and chunking."""
    
    def __init__(self):
        """Initialize document loader with configured processors and text splitter."""
        self.processors = {'.pdf': PDFProcessor()} # Register PDF processor
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS
        )
    
    def load_documents(self, file_path: Optional[Path] = None) -> List[Document]:
        """
        Load and process documents into chunks.
        
        Args:
            file_path: Path to document file. If None, uses config default.
            
        Returns:
            List of Document objects with text chunks.
        """
        if file_path is None:
            file_path = config.document_full_path
        
        if not file_path.exists():
            print(f"ERROR: File not found at {file_path}")
            return []
        
        # Get appropriate processor
        processor = self._get_processor(file_path)
        if not processor:
            print(f"ERROR: Unsupported file format: {file_path.suffix}")
            return []
        
        # Extract text
        print(f"Extracting text from {file_path.name}")
        full_text = processor.extract_text(file_path)
        
        if not full_text.strip():
            print("ERROR: No text extracted from document")
            return []
        
        print(f"Extracted {len(full_text)} characters")
        
        # Split into chunks
        text_chunks = self.text_splitter.split_text(full_text)
        print(f"Created {len(text_chunks)} text chunks")
        
        # Convert to Document objects
        documents = []
        for i, chunk in enumerate(text_chunks):
            if chunk.strip():  # Only add non-empty chunks
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "chunk_id": i,
                        "total_chunks": len(text_chunks)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _get_processor(self, file_path: Path) -> Optional[PDFProcessor]:
        """Get appropriate processor for file type."""
        return self.processors.get(file_path.suffix.lower())