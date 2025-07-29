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
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        length_function=len,
    )
    
    documents = []
    
    for page_data in pages_data:
        # Split text into chunks
        chunks = text_splitter.split_text(page_data["text"])
        
        for chunk_index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # Enhanced metadata with chunk tracking
            metadata = {
                "page": page_data["page"],
                "source": page_data["source"],
                "file_type": page_data.get("file_type", "unknown")
            }
            
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    logger.info(f"Created {len(documents)} document chunks with enhanced metadata")
    return documents


class PDFLoader:
    """Enhanced PDF loader with better content extraction."""
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load PDF with improved text and table extraction."""
        logger.info(f"Loading PDF: {file_path}")
        pages_data = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    
                    # Extract and format tables
                    tables = page.extract_tables()
                    table_text = ""
                    
                    if tables:
                        table_parts = []
                        for i, table in enumerate(tables):
                            if table and len(table) > 1:  # Skip empty or single-row tables
                                # Better table formatting
                                headers = table[0] if table[0] else []
                                rows = table[1:] if len(table) > 1 else []
                                
                                if headers:
                                    formatted_table = [f"Table {i+1}:"]
                                    formatted_table.append(" | ".join(str(h or "").strip() for h in headers))
                                    formatted_table.append("-" * 50)
                                    
                                    for row in rows:
                                        if row:
                                            formatted_table.append(" | ".join(str(cell or "").strip() for cell in row))
                                    
                                    table_parts.append("\n".join(formatted_table))
                        
                        if table_parts:
                            table_text = f"\n\n[TABLES ON PAGE {page_num}]\n" + "\n\n".join(table_parts)
                    
                    # Combine text and tables
                    full_text = text.strip()
                    if table_text:
                        full_text += table_text
                                   
                    pages_data.append({
                        "text": full_text,
                        "page": page_num,
                        "source": file_path.name,
                        "file_type": "pdf",
                    })
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
        
        logger.info(f"Loaded {len(pages_data)} pages from PDF")
        return pages_data


class TextLoader:
    """Text file loader."""
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load plain text files."""
        logger.info(f"Loading text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                return []
            
            # Simple section detection for text files
            sections = content.split('\n\n\n')  # Triple newlines as section breaks
            pages_data = []
            
            for i, section in enumerate(sections, 1):
                if section.strip():
                    pages_data.append({
                        "text": section.strip(),
                        "page": i,
                        "source": file_path.name,
                        "file_type": "text"
                    })
            
            return pages_data
        
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise


class DocxLoader:
    """Word document loader."""
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load Word documents with table extraction."""
        logger.info(f"Loading Word document: {file_path}")
        
        try:
            doc = docx.Document(str(file_path))
            
            # Extract paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            
            # Extract tables with better formatting
            table_texts = []
            
            for i, table in enumerate(doc.tables):
                if table.rows:
                    formatted_table = [f"Table {i+1}:"]
                    
                    for row_idx, row in enumerate(table.rows):
                        row_text = " | ".join(cell.text.strip() for cell in row.cells)
                        if row_idx == 0:  # Header row
                            formatted_table.append(row_text)
                            formatted_table.append("-" * len(row_text))
                        else:
                            formatted_table.append(row_text)
                    
                    table_texts.append("\n".join(formatted_table))
            
            # Combine content
            all_text = "\n\n".join(paragraphs)
            if table_texts:
                all_text += f"\n\n[TABLES]\n" + "\n\n".join(table_texts)
            
            if not all_text.strip():
                return []
            
            return [{
                "text": all_text,
                "page": 1,
                "source": file_path.name,
                "file_type": "docx",
            }]
        
        except Exception as e:
            logger.error(f"Error loading Word document {file_path}: {e}")
            raise


class DocumentLoader:
    """Simplified document loader with enhanced metadata."""
    
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
            raise ValueError(f"Unsupported file type: {ext}. Supported: {supported}")
        return self.loaders[ext]
    
    def load_and_process_document(self, file_path: Path) -> List[Document]:
        """Load and process document with metadata."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load document data
        loader = self.get_loader(file_path)
        pages_data = loader.load(file_path)
        
        # Create chunks with enhanced metadata
        documents = create_chunks(pages_data)
        
        if not documents:
            logger.warning(f"No documents created from {file_path}")
            return []
        
        logger.info(f"Successfully processed {file_path}: {len(documents)} chunks")
        return documents


class DocumentLoaderFactory:
    """Factory for supported extensions."""
    
    @staticmethod
    def supported_extensions() -> List[str]:
        return ['.pdf', '.txt', '.md', '.rst', '.docx', '.dotx']