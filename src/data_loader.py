# data_loader.py
"""
Enhanced document loading with improved metadata, error handling, and strict citation format.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pdfplumber
import pandas as pd
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from config import Config

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Enhanced document loader with strict citation format enforcement."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True  # Track character positions
        )
        self.loaded_files = []
        self.file_stats = {}
        self.processing_errors = []
    
    def load_documents(self) -> List[LangchainDocument]:
        """Load documents based on configuration settings with progress tracking."""
        logger.info("Starting document loading process")
        
        if Config.PROCESS_MULTIPLE_DOCUMENTS:
            documents = self._load_multiple_documents()
        else:
            documents = self._load_single_document()
        
        # Log summary
        total_chunks = len(documents)
        total_files = len(self.loaded_files)
        logger.info(f"Document loading completed: {total_files} files, {total_chunks} chunks")
        
        if self.processing_errors:
            logger.warning(f"Encountered {len(self.processing_errors)} processing errors")
            for error in self.processing_errors[:3]:  # Log first 3 errors
                logger.warning(f"Error: {error}")
        
        # Validate all documents have proper citation headers
        self._validate_citation_formats(documents)
        
        return documents
    
    def _validate_citation_formats(self, documents: List[LangchainDocument]):
        """Validate that all documents have proper citation headers."""
        issues = 0
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if not content:
                continue
                
            first_line = content.split('\n')[0]
            
            # Check for proper citation header format
            if not first_line.startswith('[') or ']' not in first_line:
                logger.warning(f"Document {i} missing citation header: {first_line[:50]}...")
                issues += 1
            elif 'PAGE ' in first_line.upper() or 'Table ' in first_line:
                logger.warning(f"Document {i} has bad citation format: {first_line}")
                issues += 1
            elif not any(filename in first_line for filename in self.loaded_files):
                logger.warning(f"Document {i} has unrecognized filename in header: {first_line}")
                issues += 1
        
        if issues > 0:
            logger.warning(f"Found {issues} citation format issues out of {len(documents)} documents")
        else:
            logger.info("All documents have proper citation headers")
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get comprehensive information about loaded files."""
        total_chunks = sum(stats.get("chunks", 0) for stats in self.file_stats.values())
        total_size_mb = sum(stats.get("size_mb", 0) for stats in self.file_stats.values())
        
        return {
            "loaded_files": self.loaded_files,
            "file_stats": self.file_stats,
            "total_files": len(self.loaded_files),
            "total_chunks": total_chunks,
            "total_size_mb": round(total_size_mb, 2),
            "processing_errors": self.processing_errors,
            "avg_chunks_per_file": round(total_chunks / max(len(self.loaded_files), 1), 1)
        }
    
    def _load_single_document(self) -> List[LangchainDocument]:
        """Load single document from configured path."""
        if not Config.SINGLE_DOCUMENT_PATH.exists():
            error_msg = f"Document not found: {Config.SINGLE_DOCUMENT_PATH}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
            return []
        
        documents = self._process_file(Config.SINGLE_DOCUMENT_PATH)
        logger.info(f"Loaded single document: {Config.SINGLE_DOCUMENT_PATH.name}")
        return documents
    
    def _load_multiple_documents(self) -> List[LangchainDocument]:
        """Load multiple documents from configured directory."""
        if not Config.DOCUMENTS_DIR.exists():
            error_msg = f"Documents directory not found: {Config.DOCUMENTS_DIR}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
            return []
        
        all_documents = []
        supported_files = []
        
        # Find all supported files
        for file_path in Config.DOCUMENTS_DIR.rglob("*"):
            if self._is_supported_file(file_path):
                supported_files.append(file_path)
        
        if not supported_files:
            error_msg = f"No supported files found in {Config.DOCUMENTS_DIR}"
            logger.warning(error_msg)
            self.processing_errors.append(error_msg)
            return []
        
        logger.info(f"Found {len(supported_files)} supported files")
        
        # Process each file
        for i, file_path in enumerate(supported_files, 1):
            try:
                logger.info(f"Processing file {i}/{len(supported_files)}: {file_path.name}")
                documents = self._process_file(file_path)
                all_documents.extend(documents)
                logger.info(f"✓ {file_path.name}: {len(documents)} chunks")
            except Exception as e:
                error_msg = f"Failed to process {file_path.name}: {str(e)}"
                logger.error(error_msg)
                self.processing_errors.append(error_msg)
        
        return all_documents
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file format is supported and accessible."""
        return (
            file_path.is_file() and
            file_path.suffix.lower() in Config.SUPPORTED_FORMATS and
            not file_path.name.startswith('.') and
            file_path.stat().st_size > 0  # Non-empty files only
        )
    
    def _process_file(self, file_path: Path) -> List[LangchainDocument]:
        """Process a single file with enhanced error handling and strict citation format."""
        logger.debug(f"Processing file: {file_path.name}")
        
        try:
            documents = []
            
            # Route to appropriate processor
            if file_path.suffix.lower() == '.pdf':
                documents = self._process_pdf(file_path)
            elif file_path.suffix.lower() in ['.docx', '.dotx']:
                documents = self._process_docx(file_path)
            else:
                documents = self._process_text_file(file_path)
            
            # Validate and clean documents
            valid_documents = []
            for doc in documents:
                if doc.page_content.strip():
                    # Ensure proper citation header format
                    doc = self._enforce_citation_format(doc, file_path)
                    
                    # Add file-level metadata
                    doc.metadata.update({
                        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 3),
                        "file_extension": file_path.suffix.lower(),
                        "processing_timestamp": pd.Timestamp.now().isoformat()
                    })
                    valid_documents.append(doc)
            
            # Update tracking
            self.loaded_files.append(file_path.name)
            self.file_stats[file_path.name] = {
                "chunks": len(valid_documents),
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 3),
                "file_type": file_path.suffix.lower(),
                "content_types": list(set(doc.metadata.get("content_type", "text") for doc in valid_documents))
            }
            
            logger.debug(f"✓ Processed {file_path.name}: {len(valid_documents)} valid chunks")
            return valid_documents
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
            return []
    
    def _enforce_citation_format(self, doc: LangchainDocument, file_path: Path) -> LangchainDocument:
        """Enforce proper citation format on document content."""
        content = doc.page_content.strip()
        if not content:
            return doc
        
        lines = content.split('\n')
        first_line = lines[0] if lines else ""
        
        # Determine correct header based on metadata
        page = doc.metadata.get('page')
        if page is not None:
            correct_header = f"[{file_path.name}, Page: {page}]"
        else:
            correct_header = f"[{file_path.name}]"
        
        # Check if header needs fixing
        needs_fix = False
        
        if not first_line.startswith('[') or ']' not in first_line:
            # No header at all
            needs_fix = True
            content_without_header = content
        elif any(bad_format in first_line.upper() for bad_format in ['PAGE ', 'TABLE ', 'SECTION ']):
            # Bad header format
            needs_fix = True
            content_without_header = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        elif file_path.name not in first_line:
            # Header exists but wrong filename
            needs_fix = True
            content_without_header = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        
        if needs_fix:
            # Create new content with correct header
            if content_without_header.strip():
                new_content = f"{correct_header}\n{content_without_header}"
            else:
                new_content = correct_header
            
            # Create new document with corrected content
            new_doc = LangchainDocument(
                page_content=new_content,
                metadata=doc.metadata.copy()
            )
            
            logger.debug(f"Fixed citation header: {first_line[:50]}... -> {correct_header}")
            return new_doc
        
        return doc
    
    def _process_pdf(self, file_path: Path) -> List[LangchainDocument]:
        """Process PDF with enhanced page tracking and strict citation format."""
        documents = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                logger.debug(f"Processing PDF with {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text content
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Clean and prepare text
                        cleaned_text = self._clean_text(page_text)
                        # Create content with proper citation header
                        page_content = f"[{file_path.name}, Page: {page_num}]\n{cleaned_text}"
                        
                        # Split into chunks
                        chunks = self.text_splitter.split_text(page_content)
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            # Ensure each chunk starts with proper citation header
                            if not chunk.startswith(f"[{file_path.name}"):
                                chunk = f"[{file_path.name}, Page: {page_num}]\n{chunk}"
                            
                            metadata = {
                                "source": file_path.name,
                                "page": page_num,
                                "chunk_index": chunk_idx,
                                "content_type": "text",
                                "file_type": "pdf",
                                "total_pages": len(pdf.pages)
                            }
                            documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            try:
                                # Create DataFrame and format
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df = df.dropna(how='all').fillna('')  # Clean empty rows/cells
                                
                                if not df.empty:
                                    # Format table content with proper citation header
                                    table_content = f"[{file_path.name}, Page: {page_num}]\nTable {table_idx + 1}:\n{df.to_string(index=False)}"
                                    
                                    metadata = {
                                        "source": file_path.name,
                                        "page": page_num,
                                        "table_index": table_idx,
                                        "content_type": "table",
                                        "file_type": "pdf",
                                        "table_shape": f"{df.shape[0]}x{df.shape[1]}"
                                    }
                                    documents.append(LangchainDocument(page_content=table_content, metadata=metadata))
                            except Exception as e:
                                logger.debug(f"Error processing table on page {page_num}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading PDF {file_path.name}: {e}")
            raise
        
        return documents
    
    def _process_docx(self, file_path: Path) -> List[LangchainDocument]:
        """Process DOCX file with proper citation headers."""
        documents = []
        
        try:
            doc = Document(file_path)
            
            # Process paragraphs
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            if paragraphs:
                all_text = "\n".join(paragraphs)
                cleaned_text = self._clean_text(all_text)
                
                if cleaned_text.strip():
                    # Add proper citation header
                    content_with_header = f"[{file_path.name}]\n{cleaned_text}"
                    chunks = self.text_splitter.split_text(content_with_header)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        # Ensure chunk has proper header
                        if not chunk.startswith(f"[{file_path.name}"):
                            chunk = f"[{file_path.name}]\n{chunk}"
                        
                        metadata = {
                            "source": file_path.name,
                            "chunk_index": chunk_idx,
                            "content_type": "text",
                            "file_type": "docx",
                            "paragraph_count": len(paragraphs)
                        }
                        documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
            
            # Process tables
            for table_idx, table in enumerate(doc.tables):
                try:
                    table_data = []
                    for row in table.rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        if any(cell for cell in row_data):  # Skip empty rows
                            table_data.append(row_data)
                    
                    if len(table_data) > 1:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        df = df.dropna(how='all').fillna('')
                        
                        if not df.empty:
                            # Format table content with proper citation header
                            table_content = f"[{file_path.name}]\nTable {table_idx + 1}:\n{df.to_string(index=False)}"
                            
                            metadata = {
                                "source": file_path.name,
                                "table_index": table_idx,
                                "content_type": "table",
                                "file_type": "docx",
                                "table_shape": f"{df.shape[0]}x{df.shape[1]}"
                            }
                            documents.append(LangchainDocument(page_content=table_content, metadata=metadata))
                except Exception as e:
                    logger.debug(f"Error processing table {table_idx} in {file_path.name}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path.name}: {e}")
            raise
        
        return documents
    
    def _process_text_file(self, file_path: Path) -> List[LangchainDocument]:
        """Process text files with proper citation headers."""
        documents = []
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"Could not decode {file_path.name} with any supported encoding")
        
        cleaned_content = self._clean_text(content)
        if cleaned_content.strip():
            # Add proper citation header
            content_with_header = f"[{file_path.name}]\n{cleaned_content}"
            chunks = self.text_splitter.split_text(content_with_header)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Ensure chunk has proper header
                if not chunk.startswith(f"[{file_path.name}"):
                    chunk = f"[{file_path.name}]\n{chunk}"
                
                metadata = {
                    "source": file_path.name,
                    "chunk_index": chunk_idx,
                    "content_type": "text",
                    "file_type": file_path.suffix.lower() or "txt",
                    "encoding": encoding
                }
                documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        # Remove empty lines but preserve paragraph breaks
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True
        
        return '\n'.join(cleaned_lines).strip()


def load_data() -> Tuple[List[LangchainDocument], Dict[str, Any]]:
    """Main function to load and process documents with enhanced error handling."""
    try:
        loader = DocumentLoader()
        documents = loader.load_documents()
        file_info = loader.get_file_info()
        
        # Log loading summary
        logger.info(f"Loading complete: {file_info['total_files']} files, {file_info['total_chunks']} chunks")
        if file_info['processing_errors']:
            logger.warning(f"Encountered {len(file_info['processing_errors'])} errors during loading")
        
        return documents, file_info
    
    except Exception as e:
        logger.error(f"Critical error in load_data: {e}")
        return [], {"error": str(e), "total_files": 0, "total_chunks": 0}