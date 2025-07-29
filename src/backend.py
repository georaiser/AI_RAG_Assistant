"""
Backend with improved citations and source tracking.
"""

import logging
from typing import Dict, List, Any, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import Document
from pydantic import SecretStr

from src.config import config
from src.prompts import get_system_prompt, get_summarization_prompt
from src.vector_store import get_retriever

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Query processor with source tracking."""
    
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
        """Format chat history for conversation chain."""
        history = []
        
        # Skip welcome message and process user-assistant pairs
        chat_messages = messages[1:] if len(messages) > 1 else []
        
        for i in range(0, len(chat_messages) - 1, 2):  # Exclude current user message
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
            enhanced_content = f"[Source: {source}, Page {page}]\n{doc.page_content}"
            context_parts.append(enhanced_content)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources_with_relevance(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information with relevance indicators."""
        sources = []
        
        for doc in documents:
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            
            sources.append({
                'page': page,
                'source': source,
                'preview': content_preview,
                'citation': f"Page {page} from {source}"
            })
        
        return sources
    
    def process_query(self, query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process query with improved source tracking."""
        try:
            chat_history = self._format_chat_history(messages)
            chain = self._get_chain()
            
            with get_openai_callback() as callback:
                result = chain.invoke({
                    "question": query,
                    "chat_history": chat_history
                })
                
                # Extract detailed source information
                source_docs = result.get("source_documents", [])
                sources = self._extract_sources_with_relevance(source_docs)
                
                return {
                    "answer": result["answer"],
                    "sources": sources,
                    "source_citations": [s['citation'] for s in sources],
                    "total_tokens": callback.total_tokens,
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens,
                    "total_cost_usd": callback.total_cost,
                    "documents_retrieved": len(source_docs)
                }
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I encountered an error processing your question. Please try rephrasing it.",
                "sources": [],
                "source_citations": [],
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0.0,
                "documents_retrieved": 0
            }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate document summary with citations."""
        try:
            retriever = get_retriever(k=10)  # Get more docs for comprehensive summary
            
            # Use broad queries to get diverse content
            broad_queries = [
                "main topics overview",
                "important procedures guidelines",
                "technical specifications features"
            ]
            
            all_documents = []
            for query in broad_queries:
                docs = retriever.get_relevant_documents(query)
                all_documents.extend(docs)
            
            # Remove duplicates and limit
            seen_content = set()
            unique_docs = []
            for doc in all_documents:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
                if len(unique_docs) >= 10:
                    break
            
            if not unique_docs:
                return {
                    "summary": "No content available for summarization.",
                    "total_tokens": 0,
                    "total_cost_usd": 0.0
                }
            
            # Prepare content with citations
            content_with_citations = self._enhance_context_with_citations(unique_docs)
            
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
                "summary": "Unable to generate summary at this time.",
                "total_tokens": 0,
                "total_cost_usd": 0.0
            }


# Global processor instance
_processor = QueryProcessor()

def handle_query(query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Handle query processing."""
    return _processor.process_query(query, messages)

def generate_document_summary() -> Dict[str, Any]:
    """Generate document summary."""
    return _processor.generate_summary()