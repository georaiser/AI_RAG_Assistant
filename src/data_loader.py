"""
Document loader module for the Docu Bot application.
Separated concerns with dedicated loaders for different file formats.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod

import pdfplumber
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from src.config import config

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    @abstractmethod
    def extract_text_with_metadata(self) -> List[Dict[str, Any]]:
        """Extract text with metadata from the document."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass


class PDFLoader(BaseDocumentLoader):
    """PDF loader with support for text and table extraction."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.pdf']
    
    def extract_text_with_metadata(self) -> List[Dict[str, Any]]:
        """Extract text, tables, and image info with page metadata."""
        try:
            logger.info(f"Extracting content from PDF: {self.file_path}")
            pages_data = []
            
            with pdfplumber.open(self.file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_content = self._extract_page_content(page, page_num)
                    if page_content["text"].strip():
                        pages_data.append(page_content)
                    
                    if page_num % 20 == 0:
                        logger.info(f"Processed {page_num}/{len(pdf.pages)} pages")
            
            logger.info(f"Extracted {len(pages_data)} pages with content")
            return pages_data
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            raise
    
    def _extract_page_content(self, page, page_num: int) -> Dict[str, Any]:
        """Extract all content types from a single page."""
        content = {
            "text": "",
            "page": page_num,
            "source": str(self.file_path.name),
            "tables": []
        }
        
        # Extract text
        page_text = page.extract_text()
        if page_text:
            content["text"] = page_text.strip()
        
        # Extract tables
        tables = page.extract_tables()
        if tables:
            content["tables"] = self._process_tables(tables)
            # Add table text to main content
            table_text = self._tables_to_text(tables)
            if table_text:
                content["text"] += f"\n\n[TABLES]\n{table_text}"
        
        return content
    
    def _process_tables(self, tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
        """Process extracted tables into structured format."""
        processed_tables = []
        for i, table in enumerate(tables):
            if table and len(table) > 0:
                processed_tables.append({
                    "table_id": i,
                    "rows": len(table),
                    "cols": len(table[0]) if table[0] else 0,
                    "data": table
                })
        return processed_tables
    
    def _tables_to_text(self, tables: List[List[List[str]]]) -> str:
        """Convert tables to readable text format."""
        table_texts = []
        for i, table in enumerate(tables):
            if table:
                table_text = f"Table {i+1}:\n"
                for row in table:
                    if row:
                        # Clean and join cells
                        clean_row = [cell.strip() if cell else "" for cell in row]
                        table_text += " | ".join(clean_row) + "\n"
                table_texts.append(table_text)
        return "\n".join(table_texts)


class TextLoader(BaseDocumentLoader):
    """Plain text file loader."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.txt', '.md', '.rst']
    
    def extract_text_with_metadata(self) -> List[Dict[str, Any]]:
        """Extract text from plain text files."""
        try:
            logger.info(f"Extracting text from: {self.file_path}")
            
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if not content.strip():
                return []
            
            return [{
                "text": content.strip(),
                "page": 1,  # Text files don't have pages
                "source": str(self.file_path.name),
                "file_type": "text"
            }]
            
        except Exception as e:
            logger.error(f"Error extracting text content: {str(e)}")
            raise


class DocxLoader(BaseDocumentLoader):
    """Microsoft Word document loader."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.docx', '.dotx']
    
    def extract_text_with_metadata(self) -> List[Dict[str, Any]]:
        """Extract text from Word documents with paragraph-level metadata."""
        try:
            logger.info(f"Extracting text from Word document: {self.file_path}")
            
            doc = docx.Document(str(self.file_path))
            pages_data = []
            
            # Extract paragraphs with metadata
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    para_data = {
                        "text": paragraph.text.strip(),
                        "page": i + 1,  # Use paragraph number as "page"
                        "source": str(self.file_path.name),
                        "paragraph_style": paragraph.style.name if paragraph.style else "Normal",
                        "file_type": "docx"
                    }
                    pages_data.append(para_data)
            
            # Extract tables if present
            if doc.tables:
                table_text = self._extract_tables_from_docx(doc.tables)
                if table_text:
                    pages_data.append({
                        "text": table_text,
                        "page": len(pages_data) + 1,
                        "source": str(self.file_path.name),
                        "paragraph_style": "Table",
                        "file_type": "docx"
                    })
            
            logger.info(f"Extracted {len(pages_data)} sections from Word document")
            return pages_data
            
        except Exception as e:
            logger.error(f"Error extracting Word document content: {str(e)}")
            raise
    
    def _extract_tables_from_docx(self, tables) -> str:
        """Extract tables from Word document."""
        table_texts = []
        for i, table in enumerate(tables):
            table_text = f"Table {i+1}:\n"
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                table_text += row_text + "\n"
            table_texts.append(table_text)
        return "\n".join(table_texts)


class DocumentProcessor:
    """Document processor that handles chunking and metadata enhancement."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
            length_function=len,
        )
    
    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Document]:
        """Create chunks with metadata and context preservation."""
        if not pages_data:
            return []
        
        documents = []
        
        for page_data in pages_data:
            # Split page text into chunks
            chunks = self.text_splitter.split_text(page_data["text"])
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    # Create document with metadata
                    metadata = {
                        "page": page_data["page"],
                        "source": page_data["source"],
                        "chunk_id": f"page_{page_data['page']}_chunk_{i}",
                        "total_chunks": len(chunks)
                    }
                    
                    # Add additional metadata based on file type
                    if "file_type" in page_data:
                        metadata["file_type"] = page_data["file_type"]
                    if "paragraph_style" in page_data:
                        metadata["paragraph_style"] = page_data["paragraph_style"]
                    if "tables" in page_data and page_data["tables"]:
                        metadata["has_tables"] = True
                        metadata["table_count"] = len(page_data["tables"])
                    
                    doc = Document(page_content=chunk, metadata=metadata)
                    documents.append(doc)
        
        logger.info(f"Created {len(documents)} enhanced chunks")
        return documents


class DocumentLoaderFactory:
    """Factory class to create appropriate document loaders."""
    
    _loaders = {
        '.pdf': PDFLoader,
        '.txt': TextLoader,
        '.md': TextLoader,
        '.rst': TextLoader,
        '.docx': DocxLoader,
        '.dotx': DocxLoader,
    }
    
    @classmethod
    def create_loader(cls, file_path: Path) -> BaseDocumentLoader:
        """Create appropriate loader based on file extension."""
        file_extension = file_path.suffix.lower()
        
        if file_extension not in cls._loaders:
            supported = ', '.join(cls._loaders.keys())
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: {supported}")
        
        loader_class = cls._loaders[file_extension]
        return loader_class(file_path)
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return list of all supported file extensions."""
        return list(cls._loaders.keys())


# function to load and process documents
def load_and_process_document(file_path: Path) -> List[Document]:
    """Load and process any supported document type."""
    try:
        # Create appropriate loader
        loader = DocumentLoaderFactory.create_loader(file_path)
        
        # Extract content with metadata
        pages_data = loader.extract_text_with_metadata()
        
        # Process into chunks
        processor = DocumentProcessor()
        documents = processor.create_chunks(pages_data)
        
        if not documents:
            logger.warning(f"No documents created from {file_path}")
            return []
        
        logger.info(f"Successfully created {len(documents)} documents from {file_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        return []


# Main function to load data with enhanced chunking and metadata
def load_data(path: Path) -> List[str]:
    """Load data with enhanced chunking and metadata."""
    try:
        docs = load_and_process_document(path)
        return [doc.page_content for doc in docs]

    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        return []