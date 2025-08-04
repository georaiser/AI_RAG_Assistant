# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) system for technical document analysis with proper citations, built following KISS principles and single responsibility design.

## Features

- **Multi-format Support**: PDF, DOCX, TXT, MD, RST files
- **Table Extraction**: Extracts and processes tables from documents
- **Proper Citations**: All responses include [Source: filename, Page: X] references
- **Document Summary**: Generate comprehensive summaries of loaded documents
- **Flexible Configuration**: Choose between single or multiple document processing
- **Usage Tracking**: Monitor token usage and costs
- **Multiple Embedding Options**: OpenAI and HuggingFace embeddings
- **Search Strategies**: Similarity, MMR, and score threshold search
- **Clean Architecture**: KISS principles with single responsibility classes

## Quick Start

1. **Clone and Install**
   ```bash
   git clone <repository>
   cd rag-document-assistant
   pip install -r requirements.txt
   ```

2. **Environment Setup**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. **Configure Documents**
   - Edit `config.py` to set document paths
   - For single document: Set `PROCESS_MULTIPLE_DOCUMENTS = False`
   - For multiple documents: Set `PROCESS_MULTIPLE_DOCUMENTS = True`

4. **Place Documents**
   - Single file mode: Put your document in `data/` folder
   - Multiple files mode: Put documents in `data/documents/` folder

5. **Run Application**
   ```bash
   streamlit run app.py
   ```

## Configuration Options

### Document Processing
```python
# Process single document
PROCESS_MULTIPLE_DOCUMENTS = False
SINGLE_DOCUMENT_PATH = Path("data/your_document.pdf")

# OR process multiple documents
PROCESS_MULTIPLE_DOCUMENTS = True
DOCUMENTS_DIR = Path("data/documents")
```

### Embedding Models

**OpenAI Options:**
- `text-embedding-3-small` (fast, good quality)
- `text-embedding-3-large` (better quality)
- `text-embedding-ada-002` (legacy, reliable)

**HuggingFace Options:**
- `sentence-transformers/all-MiniLM-L6-v2` (fast, good)
- `sentence-transformers/all-mpnet-base-v2` (better quality)
- `BAAI/bge-small-en-v1.5` (excellent for retrieval)

### Search Strategies
- **Similarity**: Basic semantic similarity search
- **MMR**: Maximum Marginal Relevance (reduces redundancy)
- **Score Threshold**: Only returns results above similarity threshold

## Project Structure

```
project/
├── src/
│   ├── config.py              # Configuration settings
│   ├── app.py                 # Main Streamlit application
│   ├── data_loader.py         # Document loading and processing
│   ├── backend.py             # RAG engine and query processing
│   ├── vector_store.py        # Vector store management
│   ├── prompts.py             # System prompts with citations
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (create this)
├── data/                  # Document storage
│   ├── your_document.pdf  # Single document mode
│   └── documents/         # Multiple documents mode
│       ├── doc1.pdf
│       ├── doc2.docx
│       └── doc3.txt
└── vector_store/          # ChromaDB storage (auto-created)
```

## Architecture

The project follows KISS principles with clear separation of concerns:

- **Configurations** : Centralized configuration management
- **Document Loader**: Handles all document format processing
- **Vector Store**   : Manages embeddings and retrieval
- **Backend Logics** : Core question-answering logic
- **Streamlit App**  : Web interface and user interaction

## Key Features

### Citation System
Every response includes proper source citations:
```
[Source: document.pdf, Page: 5]
[Table from document.pdf, Page: 3]
[Source: doc1.pdf, doc2.pdf]
```

### Table Processing
Automatically extracts and processes tables from:
- PDF files (using pdfplumber)
- DOCX files (using python-docx)
- Converts tables to readable text format

### Document Summary
Generate comprehensive summaries covering:
- Document overview and main topics
- Key concepts and methods
- Findings and results
- Table and data summaries
- Proper citations throughout

### Usage Tracking
Monitor your API usage:
- Token consumption per query
- Cost tracking in USD
- Session totals
- Real-time statistics in sidebar

## Usage Examples

### Basic Questions
```
"What is the main conclusion of this research?"
"Explain the methodology used in section 3"
"What are the key findings?"
```

### Table Queries
```
"What data is shown in Table 2?"
"Summarize the results table"
"What are the performance metrics?"
```

### Technical Queries
```
"How was the experiment designed?"
"What statistical methods were used?"
"What are the limitations mentioned?"
```

## Configuration Tips

1. **For Academic Papers**: Use MMR search strategy to reduce redundancy
2. **For Technical Docs**: Use similarity search for direct matches
3. **Large Documents**: Increase chunk size to 1500-2000
4. **Multiple Small Files**: Decrease chunk size to 800-1000

## Troubleshooting

### Common Issues

1. **Vector Store Not Found**
   - Delete `vector_store/` folder and restart app
   - Check document paths in config.py

2. **No Documents Loaded**
   - Verify file formats are supported
   - Check file size limits (default: 50MB)
   - Ensure documents are in correct directory

3. **API Errors**
   - Verify OPENAI_API_KEY in .env file
   - Check API key permissions and billing

### Performance Tips

1. **Faster Responses**: Use `text-embedding-3-small`
2. **Better Quality**: Use `text-embedding-3-large`
3. **Local Processing**: Use HuggingFace embeddings (no API costs)

## Contributing

The project follows KISS principles:
- Single responsibility for each class
- Minimal dependencies
- Clear separation of concerns
- Comprehensive error handling
- Proper logging and debugging

