# data_loader.py
"""
Document loading and processing for RAG system.
Handles multiple document formats with metadata preservation and comprehensive logging.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pdfplumber
import pandas as pd
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from config import Config

# Setup logging
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading and processing of various document formats."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.loaded_files = []
        self.file_stats = {}
    
    def load_documents(self) -> List[LangchainDocument]:
        """Load documents based on configuration settings."""
        logger.info("Starting document loading process")
        
        if Config.PROCESS_MULTIPLE_DOCUMENTS:
            documents = self._load_multiple_documents()
        else:
            documents = self._load_single_document()
        
        logger.info(f"Document loading completed. Total chunks: {len(documents)}")
        return documents
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about loaded files."""
        return {
            "loaded_files": self.loaded_files,
            "file_stats": self.file_stats,
            "total_files": len(self.loaded_files),
            "total_chunks": sum(stats.get("chunks", 0) for stats in self.file_stats.values())
        }
    
    def _load_single_document(self) -> List[LangchainDocument]:
        """Load single document from configured path."""
        if not Config.SINGLE_DOCUMENT_PATH.exists():
            logger.error(f"Document not found: {Config.SINGLE_DOCUMENT_PATH}")
            return []
        
        documents = self._process_file(Config.SINGLE_DOCUMENT_PATH)
        logger.info(f"Loaded single document: {Config.SINGLE_DOCUMENT_PATH.name}")
        return documents
    
    def _load_multiple_documents(self) -> List[LangchainDocument]:
        """Load multiple documents from configured directory."""
        if not Config.DOCUMENTS_DIR.exists():
            logger.error(f"Documents directory not found: {Config.DOCUMENTS_DIR}")
            return []
        
        all_documents = []
        supported_files = []
        
        # Find all supported files
        for file_path in Config.DOCUMENTS_DIR.rglob("*"):
            if self._is_supported_file(file_path):
                supported_files.append(file_path)
        
        logger.info(f"Found {len(supported_files)} supported files")
        
        # Process each file
        for file_path in supported_files:
            try:
                documents = self._process_file(file_path)
                all_documents.extend(documents)
                logger.info(f"Processed: {file_path.name} -> {len(documents)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
        
        return all_documents
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        is_supported = (
            file_path.is_file() and
            file_path.suffix.lower() in Config.SUPPORTED_FORMATS and
            not file_path.name.startswith('.')
        )
        
        if is_supported:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.debug(f"File {file_path.name}: {file_size_mb:.2f} MB")
        
        return is_supported
    
    def _process_file(self, file_path: Path) -> List[LangchainDocument]:
        """Process a single file and return chunked documents."""
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            content, tables = self._extract_content(file_path)
            if not content and not tables:
                logger.warning(f"No content extracted from {file_path.name}")
                return []
            
            documents = []
            
            # Process text content
            if content:
                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    metadata = self._create_metadata(file_path, i, len(chunks), "text")
                    documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
            
            # Process tables
            for j, table_content in enumerate(tables):
                metadata = self._create_metadata(file_path, j, len(tables), "table")
                documents.append(LangchainDocument(page_content=table_content, metadata=metadata))
            
            # Update tracking
            self.loaded_files.append(file_path.name)
            self.file_stats[file_path.name] = {
                "chunks": len(documents),
                "text_chunks": len(chunks) if content else 0,
                "tables": len(tables),
                "size_mb": file_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Successfully processed {file_path.name}: {len(documents)} total chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return []
    
    def _create_metadata(self, file_path: Path, index: int, total: int, content_type: str) -> Dict[str, Any]:
        """Create standardized metadata for documents."""
        metadata = {
            "source": file_path.name,
            "file_path": str(file_path),
            "content_type": content_type,
            "index": index,
            "total": total
        }
        
        if content_type == "text":
            metadata["chunk_index"] = index
            metadata["total_chunks"] = total
        elif content_type == "table":
            metadata["table_index"] = index
            metadata["total_tables"] = total
        
        return metadata
    
    def _extract_content(self, file_path: Path) -> Tuple[str, List[str]]:
        """Extract text content and tables from file."""
        extension = file_path.suffix.lower()
        
        extractors = {
            '.pdf': self._extract_from_pdf,
            '.docx': self._extract_from_docx,
            '.dotx': self._extract_from_docx,
            '.txt': self._extract_from_text,
            '.md': self._extract_from_text,
            '.rst': self._extract_from_text
        }
        
        extractor = extractors.get(extension)
        if extractor:
            return extractor(file_path)
        else:
            logger.warning(f"No extractor for file type: {extension}")
            return "", []
    
    def _extract_from_pdf(self, file_path: Path) -> Tuple[str, List[str]]:
        """Extract content from PDF file."""
        text_content = ""
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                logger.debug(f"Processing PDF with {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n[Page {page_num}]\n{page_text}\n"
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_text = f"[Table from {file_path.name}, Page {page_num}]\n{df.to_string(index=False)}"
                                tables.append(table_text)
                            except Exception as e:
                                logger.warning(f"Error processing table on page {page_num}: {e}")
        
        except Exception as e:
            logger.error(f"Error extracting from PDF {file_path.name}: {e}")
            return "", []
        
        return text_content.strip(), tables
    
    def _extract_from_docx(self, file_path: Path) -> Tuple[str, List[str]]:
        """Extract content from DOCX file."""
        try:
            doc = Document(file_path)
            text_content = ""
            tables = []
            
            # Extract paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content += paragraph.text + "\n"
            
            # Extract tables
            for table_num, table in enumerate(doc.tables):
                try:
                    table_data = []
                    for row in table.rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        table_data.append(row_data)
                    
                    if table_data and len(table_data) > 1:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        table_text = f"[Table from {file_path.name}, Table {table_num + 1}]\n{df.to_string(index=False)}"
                        tables.append(table_text)
                except Exception as e:
                    logger.warning(f"Error processing table {table_num}: {e}")
            
            return text_content.strip(), tables
        
        except Exception as e:
            logger.error(f"Error extracting from DOCX {file_path.name}: {e}")
            return "", []
    
    def _extract_from_text(self, file_path: Path) -> Tuple[str, List[str]]:
        """Extract content from text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content.strip(), []
        except Exception as e:
            logger.error(f"Error extracting from text file {file_path.name}: {e}")
            return "", []


def load_data() -> Tuple[List[LangchainDocument], Dict[str, Any]]:
    """Main function to load and process documents."""
    loader = DocumentLoader()
    documents = loader.load_documents()
    file_info = loader.get_file_info()
    return documents, file_info