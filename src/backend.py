"""
Refactored backend with advanced RAG techniques.
Simplified and removed code duplication.
"""

import logging
from typing import Dict, List, Any, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import Document
from pydantic import SecretStr

from src.config import config
from src.prompts import get_system_prompt, get_query_expansion_prompt
from src.vector_store import get_retriever

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Enhanced query processor with advanced RAG techniques."""
    
    def __init__(self):
        self.llm = self._create_llm(config.TEMPERATURE)
        self.query_llm = self._create_llm(0.05)  # Lower temperature for query expansion
        self._chain = None
    
    def _create_llm(self, temperature: float) -> ChatOpenAI:
        """Create ChatOpenAI instance with given temperature."""
        return ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=temperature,
            api_key=SecretStr(config.OPENAI_API_KEY)
        )
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms."""
        try:
            expansion_prompt = get_query_expansion_prompt()
            expansion_chain = expansion_prompt | self.query_llm
            result = expansion_chain.invoke({"query": query})
            
            # Handle different content types
            content = result.content
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            
            # Extract first valid expanded query
            expanded_queries = [
                q.strip() for q in content.split('\n') 
                if q.strip() and not q.startswith('#')
            ]
            
            return expanded_queries[0] if expanded_queries else query
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return query
    
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
                verbose=True
            )
            logger.info("RAG chain initialized")
        
        return self._chain
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Format chat history for LangChain."""
        formatted_history = []
        
        # Skip welcome message and current user message
        chat_messages = messages[1:-1]
        
        # Process in user-assistant pairs
        for i in range(0, len(chat_messages), 2):
            if i + 1 < len(chat_messages):
                user_msg = chat_messages[i]
                assistant_msg = chat_messages[i + 1]
                
                if (user_msg.get('role') == 'user' and 
                    assistant_msg.get('role') == 'assistant'):
                    formatted_history.append((
                        user_msg['content'], 
                        assistant_msg['content']
                    ))
        
        logger.info(f"Formatted {len(formatted_history)} chat history pairs")
        return formatted_history
    
    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """Extract unique source information from documents."""
        sources = set()
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                page = doc.metadata.get('page', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                sources.add(f"Page {page} from {source}")
        return list(sources)
    
    def process_query(self, query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process query with enhanced RAG techniques."""
        try:
            # Expand query and format history
            expanded_query = self._expand_query(query)
            chat_history = self._format_chat_history(messages)
            
            logger.info(f"Expanded query: '{expanded_query}'")
            logger.info(f"Using {len(chat_history)} conversation pairs")
            
            # Process with chain
            chain = self._get_chain()
            
            with get_openai_callback() as callback:
                result = chain.invoke({
                    "question": query,  # Use original query for chain
                    "chat_history": chat_history
                })
                
                return {
                    "answer": result["answer"],
                    "sources": self._extract_sources(result.get("source_documents", [])),
                    "total_tokens": callback.total_tokens,
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens,
                    "total_cost_usd": callback.total_cost,
                    "source_documents": result.get("source_documents", []),
                    "expanded_query": expanded_query
                }
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error. Please try rephrasing your question.",
                "sources": [],
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0.0,
                "source_documents": []
            }


# Global processor instance
_processor = QueryProcessor()

def handle_query(query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Handle query processing."""
    return _processor.process_query(query, messages)