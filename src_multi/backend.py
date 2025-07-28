"""
Backend with improved retrieval strategies and fallback mechanisms.
"""

import logging
from typing import Dict, List, Any, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import Document
from pydantic import SecretStr
from pathlib import Path

from src.config import config
from src.prompts import (
    get_system_prompt, 
    get_query_expansion_prompt, 
    get_fallback_query_prompt,
    get_summarization_prompt
)
from src.vector_store import get_retriever

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Qquery processor with improved retrieval and fallback strategies."""
    
    def __init__(self):
        self.llm = self._create_llm(config.TEMPERATURE)
        self.query_llm = None # LLM for query expansion
        self.summary_llm = None # LLM for summarization
        self._chain = None
    
    def _create_llm(self, temperature: float) -> ChatOpenAI:
        """Create ChatOpenAI instance with given temperature."""
        return ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=temperature,
            api_key=SecretStr(config.OPENAI_API_KEY)
        )
    
    def _expand_query(self, query: str) -> List[str]:
        """Query expansion returning multiple variations."""
        if not config.ENABLE_QUERY_EXPANSION:
            return [query]
            
        if self.query_llm is None:
            self.query_llm = self._create_llm(config.TEMPERATURE_EXPANSION)
            
        try:
            expansion_prompt = get_query_expansion_prompt()
            expansion_chain = expansion_prompt | self.query_llm
            result = expansion_chain.invoke({"query": query})
            
            # Handle result content
            content = result.content
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            
            # Extract multiple query variations
            variations = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('*'):
                    # Remove numbering if present
                    if line.startswith(('1.', '2.', '3.', '-')):
                        line = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if line:
                        variations.append(line)
            
            # Ensure we have at least the original query
            if not variations:
                variations = [query]
            elif query not in variations:
                variations.insert(0, query)
            
            return variations[:config.MAX_QUERY_VARIATIONS]
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def _get_fallback_queries(self, original_query: str) -> List[str]:
        """Generate fallback queries for when initial search fails."""
        if not config.ENABLE_FALLBACK_SEARCH:
            return []
            
        try:
            if self.query_llm is None:
                self.query_llm = self._create_llm(config.TEMPERATURE_EXPANSION)
            
            fallback_prompt = get_fallback_query_prompt()
            fallback_chain = fallback_prompt | self.query_llm
            result = fallback_chain.invoke({"original_query": original_query})
            
            content = result.content
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            
            fallback_queries = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove numbering
                    if line.startswith(('1.', '2.', '3.', '-')):
                        line = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if line:
                        fallback_queries.append(line)
            
            return fallback_queries
            
        except Exception as e:
            logger.error(f"Fallback query generation failed: {e}")
            return []
    
    def _retrieve_documents(self, queries: List[str], k: int = None) -> List[Document]:
        """Retrieve documents using multiple query strategies."""
        k = k or config.RETRIEVAL_K
        retriever = get_retriever(k)
        
        if not retriever:
            logger.error("No retriever available")
            return []
        
        all_documents = []
        seen_content = set()
        
        for query in queries:
            try:
                docs = retriever.get_relevant_documents(query)
                for doc in docs:
                    # Avoid duplicate content
                    content_hash = hash(doc.page_content[:200])  # Hash first 200 chars
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_documents.append(doc)
                
                # If we have enough good documents, we can stop
                if len(all_documents) >= k:
                    break
                    
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query}': {e}")
                continue
        
        return all_documents[:k * 2]  # Return up to 2x the requested amount for better context
    
    def _get_chain(self) -> ConversationalRetrievalChain:
        """Get or create conversational retrieval chain."""
        if self._chain is None:
            retriever = get_retriever(config.RETRIEVAL_K)
            prompt = get_system_prompt()
            
            self._chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=False
            )
            logger.info("RAG chain initialized")
        
        return self._chain
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Simplified and reliable chat history formatting."""
        formatted_history = []
        
        # Skip welcome message (index 0) and current user message (last index)
        if len(messages) < 3:  # Need at least welcome + user + assistant
            return formatted_history
        
        chat_messages = messages[1:-1]  # Skip first (welcome) and last (current query)
        
        # Process in user-assistant pairs
        for i in range(0, len(chat_messages), 2):
            if i + 1 < len(chat_messages):
                user_msg = chat_messages[i]
                assistant_msg = chat_messages[i + 1]
                
                if (user_msg.get('role') == 'user' and 
                    assistant_msg.get('role') == 'assistant'):
                    formatted_history.append((
                        user_msg['content'].strip(), 
                        assistant_msg['content'].strip()
                    ))
        
        return formatted_history
    
    def _extract_sources(self, documents: List[Document]) -> Dict[str, List[str]]:
        """Extract sources grouped by document."""
        sources_by_doc = {}
        for doc in documents:
            doc_name = doc.metadata.get('document_name', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            
            if doc_name not in sources_by_doc:
                sources_by_doc[doc_name] = []
            sources_by_doc[doc_name].append(f"Page {page}")
        
        return sources_by_doc
    
    def _has_sufficient_context(self, documents: List[Document]) -> bool:
        """Check if we have sufficient context to answer the question."""
        if not documents:
            return False
        
        total_content_length = sum(len(doc.page_content) for doc in documents)
        return total_content_length >= config.MIN_CONTEXT_LENGTH
    
    def process_query(self, query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process query with retrieval strategies."""
        try:
            chat_history = self._format_chat_history(messages)
            
            # Step 1: Try expanded queries
            query_variations = self._expand_query(query)
            documents = self._retrieve_documents(query_variations)
            
            # Step 2: If insufficient context, try fallback queries
            if not self._has_sufficient_context(documents) and config.ENABLE_FALLBACK_SEARCH:
                logger.info("Insufficient context, trying fallback queries...")
                fallback_queries = self._get_fallback_queries(query)
                if fallback_queries:
                    fallback_docs = self._retrieve_documents(fallback_queries, config.FALLBACK_K)
                    # Combine with original documents
                    all_docs = documents + fallback_docs
                    # Remove duplicates and limit
                    seen = set()
                    documents = []
                    for doc in all_docs:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen:
                            seen.add(content_hash)
                            documents.append(doc)
                        if len(documents) >= config.FALLBACK_K:
                            break
            
            # Step 3: Process with chain
            chain = self._get_chain()
            
            with get_openai_callback() as callback:
                # Use the best query variation (first one from expansion)
                query_to_use = query_variations[0] if query_variations else query
                
                result = chain.invoke({
                    "question": query_to_use,
                    "chat_history": chat_history
                })
                
                response_data = {
                    "answer": result["answer"],
                    "sources": self._extract_sources(result.get("source_documents", [])),
                    "total_tokens": callback.total_tokens,
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens,
                    "total_cost_usd": callback.total_cost,
                    "source_documents": result.get("source_documents", []),
                    "query_variations": query_variations if config.ENABLE_QUERY_EXPANSION else [query],
                    "documents_retrieved": len(documents),
                    "chat_history_pairs": len(chat_history)
                }
                
                return response_data
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing it or asking about a different aspect of the document.",
                "sources": [],
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0.0,
                "source_documents": [],
                "query_variations": [query],
                "documents_retrieved": 0,
                "chat_history_pairs": 0
            }
    
    def generate_summary(self, documents: List[Document] = None) -> Dict[str, Any]:
        """Generate document summary."""
        try:
            if self.summary_llm is None:
                self.summary_llm = self._create_llm(0.2)  # Lower temperature for summaries
            
            if not documents:
                retriever = get_retriever(k=15)  # Get more documents for comprehensive summary
                try:
                    # Use multiple broad queries for better coverage
                    broad_queries = [
                        "overview main topics key concepts",
                        "important information procedures examples",
                        "technical details specifications features"
                    ]
                    documents = self._retrieve_documents(broad_queries, k=15)
                except Exception as e:
                    logger.error(f"Failed to retrieve documents for summary: {e}")
                    return {
                        "summary": "Unable to generate summary - could not retrieve documents.",
                        "total_tokens": 0,
                        "total_cost_usd": 0.0
                    }
            
            if not documents:
                return {
                    "summary": "No documents available for summarization.",
                    "total_tokens": 0,
                    "total_cost_usd": 0.0
                }
            
            # Combine document content with better organization
            content_parts = []
            total_chars = 0
            max_chars = 12000  # Increased limit for better summaries
            
            # Sort documents by relevance/length for better content selection
            documents = sorted(documents, key=lambda x: len(x.page_content), reverse=True)
            
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content.strip()
                    if total_chars + len(content) > max_chars:
                        remaining = max_chars - total_chars
                        if remaining > 200:  # Only if substantial space left
                            content_parts.append(content[:remaining] + "...")
                        break
                    content_parts.append(content)
                    total_chars += len(content)
            
            combined_content = "\n\n---\n\n".join(content_parts)
            
            if not combined_content.strip():
                return {
                    "summary": "No content available for summarization.",
                    "total_tokens": 0,
                    "total_cost_usd": 0.0
                }
            
            # Generate summary
            summary_prompt = get_summarization_prompt()
            summary_chain = summary_prompt | self.summary_llm
            
            with get_openai_callback() as callback:
                result = summary_chain.invoke({"content": combined_content})
                
                summary_text = result.content
                if isinstance(summary_text, list):
                    summary_text = "\n".join(str(item) for item in summary_text)
                
                return {
                    "summary": summary_text.strip(),
                    "total_tokens": callback.total_tokens,
                    "total_cost_usd": callback.total_cost,
                    "documents_used": len(documents)
                }
                
        except Exception as e:
            logger.error(f"summary generation failed: {e}")
            return {
                "summary": "Sorry, I couldn't generate a summary at this time. Please try again later.",
                "total_tokens": 0,
                "total_cost_usd": 0.0
            }
        

    def generate_document_summaries(self, document_paths: List[Path] = None) -> Dict[str, Any]:
        """Generate summaries for each document separately."""
        if not document_paths:
            # Get all documents from retriever
            retriever = get_retriever(k=20)
            all_docs = retriever.get_relevant_documents("overview summary main content")
            
            # Group by document
            docs_by_name = {}
            for doc in all_docs:
                doc_name = doc.metadata.get('document_name', 'Unknown')
                if doc_name not in docs_by_name:
                    docs_by_name[doc_name] = []
                docs_by_name[doc_name].append(doc)
        
        summaries = {}
        total_cost = 0.0
        total_tokens = 0
        
        for doc_name, docs in docs_by_name.items():
            try:
                summary_result = self.generate_summary(docs)
                summaries[doc_name] = {
                    'summary': summary_result.get('summary', ''),
                    'tokens': summary_result.get('total_tokens', 0),
                    'cost': summary_result.get('total_cost_usd', 0.0),
                    'chunks_used': len(docs)
                }
                total_cost += summary_result.get('total_cost_usd', 0.0)
                total_tokens += summary_result.get('total_tokens', 0)
            except Exception as e:
                logger.error(f"Failed to summarize {doc_name}: {e}")
                summaries[doc_name] = {
                    'summary': f'Failed to generate summary: {str(e)}',
                    'tokens': 0,
                    'cost': 0.0,
                    'chunks_used': 0
                }
        
        return {
            'summaries': summaries,
            'total_tokens': total_tokens,
            'total_cost_usd': total_cost,
            'documents_processed': len(summaries)
        }
    
# Global processor instance
_processor = QueryProcessor()

def handle_query(query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Handle query processing with retrieval."""
    return _processor.process_query(query, messages)

def generate_document_summary() -> Dict[str, Any]:
    """Generate document summary."""
    return _processor.generate_summary()