# data_loader.py
"""
Document loading and processing for RAG system.
Handles multiple document formats with metadata preservation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
import pandas as pd
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from config import Config


class DocumentLoader:
    """Handles loading and processing of various document formats."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self) -> List[LangchainDocument]:
        """Load documents based on configuration settings."""
        if Config.PROCESS_MULTIPLE_DOCUMENTS:
            return self._load_multiple_documents()
        else:
            return self._load_single_document()
    
    def _load_single_document(self) -> List[LangchainDocument]:
        """Load single document from configured path."""
        if not Config.SINGLE_DOCUMENT_PATH.exists():
            print(f"ERROR: Document not found: {Config.SINGLE_DOCUMENT_PATH}")
            return []
        
        documents = self._process_file(Config.SINGLE_DOCUMENT_PATH)
        print(f"Loaded single document: {len(documents)} chunks")
        return documents
    
    def _load_multiple_documents(self) -> List[LangchainDocument]:
        """Load multiple documents from configured directory."""
        if not Config.DOCUMENTS_DIR.exists():
            print(f"ERROR: Documents directory not found: {Config.DOCUMENTS_DIR}")
            return []
        
        all_documents = []
        for file_path in Config.DOCUMENTS_DIR.rglob("*"):
            if self._is_supported_file(file_path):
                documents = self._process_file(file_path)
                all_documents.extend(documents)
        
        print(f"Loaded {len(all_documents)} chunks from multiple documents")
        return all_documents
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return (file_path.suffix.lower() in Config.SUPPORTED_FORMATS and 
                file_path.stat().st_size < Config.MAX_FILE_SIZE_MB * 1024 * 1024)
    
    def _process_file(self, file_path: Path) -> List[LangchainDocument]:
        """Process a single file and return chunked documents."""
        try:
            content, tables = self._extract_content(file_path)
            if not content:
                return []
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": str(file_path.name),
                    "file_path": str(file_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
            
            # Add table content as separate documents
            for j, table in enumerate(tables):
                metadata = {
                    "source": str(file_path.name),
                    "file_path": str(file_path),
                    "content_type": "table",
                    "table_index": j
                }
                documents.append(LangchainDocument(page_content=table, metadata=metadata))
            
            return documents
            
        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            return []
    
    def _extract_content(self, file_path: Path) -> tuple[str, List[str]]:
        """Extract text content and tables from file."""
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif extension in ['.docx', '.dotx']:
            return self._extract_from_docx(file_path)
        elif extension in ['.txt', '.md', '.rst']:
            return self._extract_from_text(file_path)
        else:
            return "", []
    
    def _extract_from_pdf(self, file_path: Path) -> tuple[str, List[str]]:
        """Extract content from PDF file."""
        text_content = ""
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text_content += f"[Page {page_num}]\n{page_text}\n\n"
                
                # Extract tables
                page_tables = page.extract_tables()
                for table_num, table in enumerate(page_tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_text = f"[Table from {file_path.name}, Page {page_num}]\n{df.to_string(index=False)}"
                        tables.append(table_text)
        
        return text_content, tables
    
    def _extract_from_docx(self, file_path: Path) -> tuple[str, List[str]]:
        """Extract content from DOCX file."""
        doc = Document(file_path)
        text_content = ""
        tables = []
        
        # Extract paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content += paragraph.text + "\n"
        
        # Extract tables
        for table_num, table in enumerate(doc.tables):
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            
            if table_data:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                table_text = f"[Table from {file_path.name}]\n{df.to_string(index=False)}"
                tables.append(table_text)
        
        return text_content, tables
    
    def _extract_from_text(self, file_path: Path) -> tuple[str, List[str]]:
        """Extract content from text files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content, []


def load_data() -> List[LangchainDocument]:
    """Main function to load and process documents."""
    loader = DocumentLoader()
    return loader.load_documents()