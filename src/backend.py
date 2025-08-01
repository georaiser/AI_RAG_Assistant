"""
Backend with improved RAG and better answer accuracy.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import Document
from pydantic import SecretStr

from src.config import config
from src.prompts import get_system_prompt, get_summarization_prompt
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Global vector store manager instance
_vector_manager = VectorStoreManager()

def get_retriever(k: Optional[int] = None):
    """Get retriever from global vector store manager."""
    return _vector_manager.get_retriever(k)


class QueryProcessor:
    """Query processor with improved retrieval and accuracy."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            api_key=SecretStr(config.OPENAI_API_KEY)
        )
        self._chain = None
    
    def _get_chain(self) -> ConversationalRetrievalChain:
        """Get or create conversational retrieval chain."""
        if self._chain is None:
            retriever = get_retriever(config.RETRIEVAL_K)
            
            if retriever is None:
                logger.error("Cannot create chain: retriever is None")
                raise ValueError("Retriever not available. Vector store may not be properly initialized.")
            
            prompt = get_system_prompt()
            
            self._chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                combine_docs_chain_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context"
                },
                return_source_documents=True,
                verbose=False
            )
            logger.info("RAG chain initialized successfully")
        
        return self._chain
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Format chat history for conversation chain."""
        history = []
        
        # Skip welcome message and current user message
        chat_messages = messages[1:-1] if len(messages) > 2 else []
        
        # Process user-assistant pairs
        for i in range(0, len(chat_messages), 2):
            if i + 1 < len(chat_messages):
                user_msg = chat_messages[i]
                assistant_msg = chat_messages[i + 1]
                
                if (user_msg.get('role') == 'user' and 
                    assistant_msg.get('role') == 'assistant'):
                    history.append((user_msg['content'], assistant_msg['content']))
        
        return history
    
    def _enhance_context_with_citations(self, documents: List[Document]) -> str:
        """Enhance context with clear page information for citations."""
        context_parts = []
        
        for i, doc in enumerate(documents):
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Document')
            
            # Add clear page markers to help with citations
            enhanced_content = f"[Page {page} from {source}]\n{doc.page_content}\n[End of Page {page}]"
            context_parts.append(enhanced_content)
        
        return "\n\n".join(context_parts)
    
    def _extract_sources_with_details(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information with better structure for citations."""
        sources = []
        seen_pages = set()
        
        for doc in documents:
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            chunk_index = doc.metadata.get('chunk_index', 0)
            
            # Create unique identifier
            page_key = f"{source}-{page}-{chunk_index}"
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            
            # Enhanced preview with context
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            sources.append({
                'page': page,
                'source': source,
                'preview': content_preview,
                'citation': f"Page {page}",
                'chunk_index': chunk_index,
                'full_reference': f"Page {page} from {source}"
            })
        
        return sources
    
    def process_query(self, query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process query with improved citation support."""
        try:
            # Check if vector store is properly initialized
            if not _vector_manager.is_initialized():
                logger.error("Vector store not initialized for query processing")
                return {
                    "answer": "The document processing system is not ready. Please wait for initialization to complete or restart the application.",
                    "sources": [],
                    "source_citations": [],
                    "full_references": [],
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_cost_usd": 0.0,
                    "documents_retrieved": 0
                }
            
            chat_history = self._format_chat_history(messages)
            chain = self._get_chain()
            
            with get_openai_callback() as callback:
                result = chain.invoke({
                    "question": query,
                    "chat_history": chat_history
                })
                
                source_docs = result.get("source_documents", [])
                sources = self._extract_sources_with_details(source_docs)
                
                return {
                    "answer": result["answer"],
                    "sources": sources,
                    "source_citations": [s['citation'] for s in sources],
                    "full_references": [s['full_reference'] for s in sources],
                    "total_tokens": callback.total_tokens,
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens,
                    "total_cost_usd": callback.total_cost,
                    "documents_retrieved": len(source_docs)
                }
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I encountered an error processing your question. Please try rephrasing it or restart the application if the problem persists.",
                "sources": [],
                "source_citations": [],
                "full_references": [],
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0.0,
                "documents_retrieved": 0
            }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate document summary with citations."""
        try:
            # Check if vector store is properly initialized
            if not _vector_manager.is_initialized():
                logger.error("Vector store not initialized for summary generation")
                return {
                    "summary": "The document processing system is not ready. Please wait for initialization to complete or restart the application.",
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "documents_used": 0
                }
            
            retriever = get_retriever(k=15)  # Get more docs for comprehensive summary
            
            if retriever is None:
                logger.error("Cannot generate summary: retriever is None")
                return {
                    "summary": "Unable to access document content for summarization.",
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "documents_used": 0
                }
            
            # Use specific queries relevant to documents
            broad_queries = [
                "main topics overview",
                "key concepts definitions", 
                "procedures methods",
                "technical specifications",
                "important guidelines",
                "examples applications"
            ]
            
            all_documents = []
            for query in broad_queries:
                try:
                    docs = retriever.get_relevant_documents(query)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Error retrieving documents for query '{query}': {e}")
                    continue
            
            # Remove duplicates based on content similarity
            unique_docs = self._deduplicate_documents(all_documents)
            
            if not unique_docs:
                return {
                    "summary": "No content available for summarization.",
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "documents_used": 0
                }
            
            # Prepare content for summarization with citations
            content_with_citations = self._prepare_summary_content(unique_docs)
            
            # Generate summary
            summary_prompt = get_summarization_prompt()
            summary_chain = summary_prompt | self.llm
            
            with get_openai_callback() as callback:
                result = summary_chain.invoke({"content": content_with_citations})
                
                return {
                    "summary": result.content,
                    "total_tokens": callback.total_tokens,
                    "total_cost_usd": callback.total_cost,
                    "documents_used": len(unique_docs),
                    "pages_covered": len(set(doc.metadata.get('page', 0) for doc in unique_docs))
                }
                
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                "summary": "Unable to generate summary at this time. Please try again or restart the application.",
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "documents_used": 0
            }
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity."""
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            # Use first 100 chars as hash to identify similar content
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
            if len(unique_docs) >= 12:
                break
        
        return unique_docs
    
    def _prepare_summary_content(self, documents: List[Document]) -> str:
        """Prepare content for summarization with proper citations."""
        content_parts = []
        for doc in documents:
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Document')
            content_parts.append(f"[Page {page}] {doc.page_content}")
        
        return "\n\n".join(content_parts)


# Global processor instance
_processor = QueryProcessor()

def handle_query(query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Handle query processing with citation support."""
    return _processor.process_query(query, messages)

def generate_document_summary() -> Dict[str, Any]:
    """Generate document summary with citations."""
    return _processor.generate_summary()

def set_vector_manager(manager: VectorStoreManager):
    """Set the global vector store manager."""
    global _vector_manager
    _vector_manager = manager