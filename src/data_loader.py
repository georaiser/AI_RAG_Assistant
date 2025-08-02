# data_loader.py
"""
Improved document loading with better page tracking and metadata.
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
        
        for file_path in Config.DOCUMENTS_DIR.rglob("*"):
            if self._is_supported_file(file_path):
                supported_files.append(file_path)
        
        logger.info(f"Found {len(supported_files)} supported files")
        
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
        return (
            file_path.is_file() and
            file_path.suffix.lower() in Config.SUPPORTED_FORMATS and
            not file_path.name.startswith('.')
        )
    
    def _process_file(self, file_path: Path) -> List[LangchainDocument]:
        """Process a single file and return chunked documents."""
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            documents = []
            
            if file_path.suffix.lower() == '.pdf':
                documents = self._process_pdf(file_path)
            elif file_path.suffix.lower() in ['.docx', '.dotx']:
                documents = self._process_docx(file_path)
            else:
                documents = self._process_text_file(file_path)
            
            # Update tracking
            self.loaded_files.append(file_path.name)
            self.file_stats[file_path.name] = {
                "chunks": len(documents),
                "size_mb": file_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Successfully processed {file_path.name}: {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return []
    
    def _process_pdf(self, file_path: Path) -> List[LangchainDocument]:
        """Process PDF with proper page tracking."""
        documents = []
        
        with pdfplumber.open(file_path) as pdf:
            logger.info(f"Processing PDF with {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # Add page marker to content for better tracking
                    page_content = f"[PAGE {page_num}]\n{page_text}"
                    
                    # Split page content into chunks
                    chunks = self.text_splitter.split_text(page_content)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        metadata = {
                            "source": file_path.name,
                            "page": page_num,
                            "chunk_index": chunk_idx,
                            "content_type": "text",
                            "file_type": "pdf"
                        }
                        documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
                        logger.debug(f"Created chunk {chunk_idx} for page {page_num}")
                
                # Process tables on this page
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:
                        try:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            table_content = f"[TABLE FROM PAGE {page_num}]\nTable {table_idx + 1}:\n{df.to_string(index=False)}"
                            
                            metadata = {
                                "source": file_path.name,
                                "page": page_num,
                                "table_index": table_idx,
                                "content_type": "table",
                                "file_type": "pdf"
                            }
                            documents.append(LangchainDocument(page_content=table_content, metadata=metadata))
                            logger.debug(f"Created table {table_idx} for page {page_num}")
                        except Exception as e:
                            logger.warning(f"Error processing table on page {page_num}: {e}")
        
        return documents
    
    def _process_docx(self, file_path: Path) -> List[LangchainDocument]:
        """Process DOCX file."""
        documents = []
        doc = Document(file_path)
        
        # Process paragraphs
        all_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if all_text.strip():
            chunks = self.text_splitter.split_text(all_text)
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "source": file_path.name,
                    "chunk_index": chunk_idx,
                    "content_type": "text"
                }
                documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
        
        # Process tables
        for table_idx, table in enumerate(doc.tables):
            try:
                table_data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                if table_data and len(table_data) > 1:
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    table_content = f"Table {table_idx + 1}:\n{df.to_string(index=False)}"
                    
                    metadata = {
                        "source": file_path.name,
                        "table_index": table_idx,
                        "content_type": "table"
                    }
                    documents.append(LangchainDocument(page_content=table_content, metadata=metadata))
            except Exception as e:
                logger.warning(f"Error processing table {table_idx}: {e}")
        
        return documents
    
    def _process_text_file(self, file_path: Path) -> List[LangchainDocument]:
        """Process text files."""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if content.strip():
            chunks = self.text_splitter.split_text(content)
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "source": file_path.name,
                    "chunk_index": chunk_idx,
                    "content_type": "text"
                }
                documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
        
        return documents


def load_data() -> Tuple[List[LangchainDocument], Dict[str, Any]]:
    """Main function to load and process documents."""
    loader = DocumentLoader()
    documents = loader.load_documents()
    file_info = loader.get_file_info()
    return documents, file_info