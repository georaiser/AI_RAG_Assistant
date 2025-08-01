"""
Document loader with metadata and chunk tracking.
"""
import logging
from typing import List, Dict, Any
from pathlib import Path
import pdfplumber
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from src.config import config

logger = logging.getLogger(__name__)


def create_chunks(pages_data: List[Dict[str, Any]]) -> List[Document]:
    """Create document chunks with metadata."""
    if not pages_data:
        logger.warning("No pages data provided for chunking")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        length_function=len,
    )
    
    documents = []
    
    for page_data in pages_data:
        page_text = page_data.get("text", "")
        if not page_text or not page_text.strip():
            logger.debug(f"Skipping empty page {page_data.get('page', 'unknown')}")
            continue
            
        try:
            # Split text into chunks
            chunks = text_splitter.split_text(page_text)
            
            for chunk_index, chunk in enumerate(chunks):
                if not chunk or not chunk.strip():
                    continue
                
                # Enhanced metadata with chunk tracking
                metadata = {
                    "page": page_data.get("page", 1),
                    "source": page_data.get("source", "unknown"),
                    "file_type": page_data.get("file_type", "unknown"),
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks)
                }
                
                documents.append(Document(page_content=chunk.strip(), metadata=metadata))
                
        except Exception as e:
            logger.error(f"Error chunking page {page_data.get('page', 'unknown')}: {e}")
            continue
    
    logger.info(f"Created {len(documents)} document chunks from {len(pages_data)} pages")
    return documents


class PDFLoader:
    """Enhanced PDF loader with better content extraction."""
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load PDF with improved text and table extraction."""
        pages_data = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing PDF with {total_pages} pages: {file_path.name}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text with better error handling
                        text = ""
                        try:
                            text = page.extract_text() or ""
                        except Exception as e:
                            logger.warning(f"Failed to extract text from page {page_num}: {e}")
                            text = ""
                        
                        # Skip completely empty pages but log them
                        if not text.strip():
                            logger.debug(f"Page {page_num} is empty, skipping")
                            continue
                        
                        # Extract and format tables
                        table_text = self._extract_tables(page, page_num)
                        
                        # Combine text and tables
                        full_text = text.strip()
                        if table_text:
                            full_text += f"\n\n{table_text}"
                        
                        # Clean up text
                        full_text = self._clean_text(full_text)
                        
                        if full_text.strip():  # Only add if there's actual content
                            pages_data.append({
                                "text": full_text,
                                "page": page_num,
                                "source": file_path.name,
                                "file_type": "pdf",
                            })
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page_num} in {file_path.name}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise ValueError(f"Failed to load PDF: {str(e)}")
        
        if not pages_data:
            raise ValueError(f"No readable content found in PDF: {file_path}")
        
        logger.info(f"Successfully extracted content from {len(pages_data)} pages")
        return pages_data
    
    def _extract_tables(self, page, page_num: int) -> str:
        """Extract and format tables from a page."""
        try:
            tables = page.extract_tables()
            if not tables:
                return ""
            
            table_parts = []
            for i, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                
                # Better table processing
                headers = table[0] if table[0] else []
                rows = table[1:] if len(table) > 1 else []
                
                if headers:
                    formatted_table = [f"Table {i+1} (Page {page_num}):"]
                    
                    # Clean headers
                    clean_headers = []
                    for h in headers:
                        header_text = str(h or "").strip()
                        if header_text:
                            clean_headers.append(header_text)
                        else:
                            clean_headers.append("Col")
                    
                    if clean_headers:
                        formatted_table.append(" | ".join(clean_headers))
                        formatted_table.append("-" * min(80, len(" | ".join(clean_headers))))
                        
                        for row in rows:
                            if row:
                                clean_row = []
                                for cell in row:
                                    cell_text = str(cell or "").strip()
                                    clean_row.append(cell_text)
                                
                                if any(clean_row):  # Only add if row has content
                                    formatted_table.append(" | ".join(clean_row))
                        
                        if len(formatted_table) > 3:  # Has headers + separator + at least one row
                            table_parts.append("\n".join(formatted_table))
            
            return f"\n[TABLES ON PAGE {page_num}]\n" + "\n\n".join(table_parts) if table_parts else ""
            
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:  # Only keep non-empty lines
                lines.append(line)
        
        # Join with single newlines and normalize spacing
        cleaned = '\n'.join(lines)
        
        # Remove excessive spaces
        import re
        cleaned = re.sub(r' +', ' ', cleaned)
        
        return cleaned


class TextLoader:
    """Text file loader with better section handling."""
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load plain text files with section detection."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read().strip()
                    logger.debug(f"Successfully read {file_path.name} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError(f"Could not decode text file with any supported encoding: {file_path}")
            
            if not content:
                raise ValueError(f"Text file is empty: {file_path}")
            
            sections = self._detect_sections(content)
            pages_data = []
            
            for i, section in enumerate(sections, 1):
                section_text = section.strip()
                if section_text:
                    pages_data.append({
                        "text": section_text,
                        "page": i,
                        "source": file_path.name,
                        "file_type": file_path.suffix.lower() or "text"
                    })
            
            if not pages_data:
                # Fallback: treat entire content as one page
                pages_data.append({
                    "text": content,
                    "page": 1,
                    "source": file_path.name,
                    "file_type": file_path.suffix.lower() or "text"
                })
            
            logger.info(f"Loaded {len(pages_data)} sections from text file")
            return pages_data
        
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise ValueError(f"Failed to load text file: {str(e)}")
    
    def _detect_sections(self, content: str) -> List[str]:
        """Detect logical sections in text content."""
        # Try different separators in order of preference
        separators = [
            '\n\n\n\n',  # Four newlines
            '\n\n\n',    # Three newlines  
            '\n---\n',   # Markdown horizontal rule
            '\n===\n',   # Alternative horizontal rule
            '\n***\n',   # Alternative separator
            '\n\n'       # Double newlines (paragraphs)
        ]
        
        for separator in separators:
            if separator in content:
                sections = content.split(separator)
                if len(sections) > 1:
                    # Filter out very short sections
                    valid_sections = [s for s in sections if s.strip() and len(s.strip()) > 50]
                    if valid_sections:
                        return valid_sections
        
        # No good separator found, return whole content
        return [content]


class DocxLoader:
    """Word document loader with improved table handling."""
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load Word documents with enhanced content extraction."""
        try:
            doc = docx.Document(str(file_path))
            
            # Extract paragraphs with better filtering
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text and len(text) > 5:  # Filter out very short paragraphs
                    paragraphs.append(text)
            
            # Extract tables
            table_texts = self._extract_docx_tables(doc)
            
            # Combine content intelligently
            all_content = []
            
            if paragraphs:
                all_content.append("DOCUMENT TEXT:")
                all_content.extend(paragraphs)
            
            if table_texts:
                all_content.append("\nDOCUMENT TABLES:")
                all_content.extend(table_texts)
            
            all_text = "\n\n".join(all_content) if all_content else ""
            
            if not all_text.strip():
                raise ValueError(f"No readable content found in Word document: {file_path}")
            
            logger.info(f"Extracted {len(paragraphs)} paragraphs and {len(table_texts)} tables from Word document")
            
            return [{
                "text": all_text,
                "page": 1,
                "source": file_path.name,
                "file_type": "docx",
            }]
            
        except Exception as e:
            logger.error(f"Error loading Word document {file_path}: {e}")
            raise ValueError(f"Failed to load Word document: {str(e)}")
    
    def _extract_docx_tables(self, doc) -> List[str]:
        """Extract and format tables from Word document."""
        table_texts = []
        
        for i, table in enumerate(doc.tables):
            if not table.rows:
                continue
            
            try:
                rows_data = []
                
                for row_idx, row in enumerate(table.rows):
                    if not row.cells:
                        continue
                    
                    cells_text = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        cells_text.append(cell_text)
                    
                    # Only add rows with actual content
                    if any(cell.strip() for cell in cells_text):
                        rows_data.append(cells_text)
                
                if rows_data:
                    formatted_table = [f"\nTable {i+1}:"]
                    
                    # Add header row with separator
                    if rows_data:
                        header_row = " | ".join(rows_data[0])
                        formatted_table.append(header_row)
                        formatted_table.append("-" * min(80, len(header_row)))
                        
                        # Add data rows
                        for row_data in rows_data[1:]:
                            data_row = " | ".join(row_data)
                            if data_row.strip():
                                formatted_table.append(data_row)
                    
                    if len(formatted_table) > 2:  # Has content beyond header
                        table_texts.append("\n".join(formatted_table))
                    
            except Exception as e:
                logger.error(f"Error processing table {i+1}: {e}")
                continue
        
        return table_texts


class DocumentLoader:
    """Main document loader with enhanced error handling."""
    
    def __init__(self):
        self.loaders = {
            '.pdf': PDFLoader(),
            '.txt': TextLoader(),
            '.md': TextLoader(),
            '.rst': TextLoader(),
            '.docx': DocxLoader(),
            '.dotx': DocxLoader()
        }
    
    def get_loader(self, file_path: Path):
        """Get appropriate loader for file type."""
        ext = file_path.suffix.lower()
        if ext not in self.loaders:
            supported = list(self.loaders.keys())
            raise ValueError(f"Unsupported file type: {ext}. Supported formats: {supported}")
        return self.loaders[ext]
    
    def load_and_process_document(self, file_path: Path) -> List[Document]:
        """Load and process document with comprehensive validation."""
        file_path = Path(file_path)
        
        # Pre-validation
        validation_result = DocumentLoaderFactory.validate_file(file_path)
        if not validation_result["valid"]:
            error_msg = "; ".join(validation_result["errors"])
            raise ValueError(f"File validation failed: {error_msg}")
        
        # Log warnings
        for warning in validation_result.get("warnings", []):
            logger.warning(warning)
        
        try:
            # Load document data
            loader = self.get_loader(file_path)
            logger.info(f"Loading document: {file_path.name}")
            pages_data = loader.load(file_path)
            
            if not pages_data:
                raise ValueError(f"No content extracted from document: {file_path}")
            
            # Create chunks with enhanced metadata
            documents = create_chunks(pages_data)
            
            if not documents:
                raise ValueError(f"No valid chunks created from document: {file_path}")
            
            # Final validation of documents
            valid_documents = self._validate_documents(documents)
            
            if not valid_documents:
                raise ValueError(f"No valid documents after final validation: {file_path}")
            
            logger.info(f"Successfully processed {file_path.name}: {len(valid_documents)} chunks from {len(pages_data)} pages")
            return valid_documents
            
        except Exception as e:
            logger.error(f"Failed to load and process document {file_path}: {e}")
            raise
    
    def _validate_documents(self, documents: List[Document]) -> List[Document]:
        """Validate processed documents."""
        valid_docs = []
        
        for doc in documents:
            # Check content length
            if not doc.page_content or len(doc.page_content.strip()) < 10:
                continue
            
            # Check metadata
            if not hasattr(doc, 'metadata') or not doc.metadata:
                logger.warning("Document missing metadata, adding defaults")
                doc.metadata = {"page": 1, "source": "unknown", "file_type": "unknown", "chunk_index": 0}
            
            # Ensure required fields
            required_fields = ["page", "source", "file_type", "chunk_index"]
            for field in required_fields:
                if field not in doc.metadata:
                    doc.metadata[field] = "unknown" if field in ["source", "file_type"] else 0
            
            valid_docs.append(doc)
        
        logger.info(f"Validated {len(valid_docs)} out of {len(documents)} documents")
        return valid_docs


class DocumentLoaderFactory:
    """Factory for supported extensions and validation."""
    
    @staticmethod
    def supported_extensions() -> List[str]:
        return ['.pdf', '.txt', '.md', '.rst', '.docx', '.dotx']
    
    @staticmethod
    def is_supported(file_path: Path) -> bool:
        """Check if file type is supported."""
        return file_path.suffix.lower() in DocumentLoaderFactory.supported_extensions()
    
    @staticmethod
    def validate_file(file_path: Path) -> Dict[str, Any]:
        """Validate file before processing with enhanced checks."""
        file_path = Path(file_path)
        
        result = {
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        # Basic existence checks
        if not file_path.exists():
            result["errors"].append(f"File does not exist: {file_path}")
            return result
        
        if not file_path.is_file():
            result["errors"].append(f"Path is not a file: {file_path}")
            return result
        
        # Size checks
        try:
            file_size = file_path.stat().st_size
            if file_size == 0:
                result["errors"].append(f"File is empty: {file_path}")
                return result
            
            size_mb = file_size / (1024 * 1024)
            if size_mb > config.MAX_FILE_SIZE_MB:
                result["errors"].append(f"File too large: {size_mb:.1f}MB > {config.MAX_FILE_SIZE_MB}MB")
                return result
            elif size_mb > config.WARN_FILE_SIZE_MB:
                result["warnings"].append(f"Large file size: {size_mb:.1f}MB. Processing may take longer.")
        
        except Exception as e:
            result["errors"].append(f"Cannot access file stats: {e}")
            return result
        
        # Format support check
        if not DocumentLoaderFactory.is_supported(file_path):
            supported = DocumentLoaderFactory.supported_extensions()
            result["errors"].append(f"Unsupported file type: {file_path.suffix}. Supported: {supported}")
            return result
        
        # File-specific validation
        if file_path.suffix.lower() == '.pdf':
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    if len(pdf.pages) == 0:
                        result["errors"].append("PDF has no pages")
                        return result
            except Exception as e:
                result["errors"].append(f"Cannot read PDF file: {e}")
                return result
        
        result["valid"] = True
        return result