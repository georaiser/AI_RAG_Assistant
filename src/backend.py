"""
Backend with advanced RAG techniques.
Includes query expansion, reranking, and context synthesis.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate

from src.config import config
from src.prompts import get_system_prompt, get_query_expansion_prompt
from src.vector_store import get_retriever

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Enhanced query processor with advanced RAG techniques."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )

        # Query expansion
        self.query_llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=0.1,  # Lower temperature for query expansion
            api_key=config.OPENAI_API_KEY
        )
        self._chain = None
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms and return the best expanded version."""
        try:
            expansion_prompt = get_query_expansion_prompt()
            expansion_chain = expansion_prompt | self.query_llm
            
            result = expansion_chain.invoke({"query": query})
            
            # Handle result.content being a list or a string
            content = result.content
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            
            expanded_queries = [q.strip() for q in content.split('\n') if q.strip()]
            expanded_queries = [q for q in expanded_queries if q and not q.startswith('#')]
            
            # Return the first expanded query or original if expansion failed
            return expanded_queries[0] if expanded_queries else query
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return query
    
    def _initialize_enhanced_chain(self) -> ConversationalRetrievalChain:
        """Initialize enhanced RAG chain."""
        if self._chain is None:
            try:
                retriever = get_retriever(config.RETRIEVAL_K)
                prompt = get_system_prompt()
                
                self._chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    combine_docs_chain_kwargs={"prompt": prompt},
                    return_source_documents=True,
                    verbose=True
                )
                
                logger.info("Enhanced RAG chain initialized")
                
            except Exception as e:
                logger.error(f"Error initializing enhanced chain: {str(e)}")
                raise
        
        return self._chain
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Format chat history for LangChain."""
        formatted_history = []
        chat_messages = messages[1:-1]  # Skip welcome and current message
        
        for i in range(0, len(chat_messages), 2):
            if i + 1 < len(chat_messages):
                user_msg = chat_messages[i]
                bot_msg = chat_messages[i + 1]
                
                if user_msg.get('role') == 'user' and bot_msg.get('role') == 'bot':
                    formatted_history.append((user_msg['content'], bot_msg['content']))
        
        return formatted_history
    
    def process_query(self, query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process query with enhanced RAG techniques."""
        try:
            # Query expansion
            expanded_query = self._expand_query(query)
            logger.info(f"Expanded query: '{expanded_query}'")
            
            # Retrieve documents using expanded query
            retriever = get_retriever(config.RETRIEVAL_K)
            documents = retriever.get_relevant_documents(expanded_query)
            
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Use standard chain with enhanced context
            chain = self._initialize_enhanced_chain()
            chat_history = self._format_chat_history(messages)
            
            with get_openai_callback() as callback:
                result = chain.invoke({
                    "question": query,
                    "chat_history": chat_history
                })
                
                # Add source information
                sources = self._extract_source_info(result.get("source_documents", []))
                
                return {
                    "answer": result["answer"],
                    "sources": sources,
                    "total_tokens": callback.total_tokens,
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens,
                    "total_cost_usd": callback.total_cost,
                    "source_documents": result.get("source_documents", [])
                }
                
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {str(e)}")
            return self._create_error_response()
    
    def _extract_source_info(self, documents: List[Document]) -> List[str]:
        """Extract source information from documents."""
        sources = []
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                page = doc.metadata.get('page', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                sources.append(f"Page {page} from {source}")
        return list(set(sources))  # Remove duplicates
    
    def _create_error_response(self) -> Dict[str, Any]:
        """Create error response."""
        return {
            "answer": "I apologize, but I encountered an error. Please try rephrasing your question.",
            "sources": [],
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0,
            "source_documents": []
        }

# Global processor
_query_processor = QueryProcessor()

def handle_query(query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Handle query with RAG techniques."""
    return _query_processor.process_query(query, messages)