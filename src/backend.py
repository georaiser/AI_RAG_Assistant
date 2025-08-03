# backend.py
"""
Simplified RAG Engine with better error handling.
"""

import logging
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from vector_store import VectorStoreManager
from prompts import SYSTEM_TEMPLATE, SUMMARY_TEMPLATE
from config import Config
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class RAGEngine:
    """Simplified RAG engine for document question-answering."""
    
    def __init__(self, vector_manager: VectorStoreManager):
        """Initialize RAG engine."""
        self.vector_manager = vector_manager
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=SecretStr(Config.OPENAI_API_KEY)
        )
        logger.info(f"RAG Engine initialized with {Config.MODEL_NAME}")
    
    def process_query(self, query: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Process user query and return response."""
        try:
            # Check if vector store is ready
            if not self.vector_manager.is_ready():
                return self._error_response("Vector store not ready")
            
            with get_openai_callback() as cb:
                # Get retriever and search documents
                retriever = self.vector_manager.get_retriever()
                if not retriever:
                    return self._error_response("Could not create retriever")
                
                docs = retriever.invoke(query)
                
                if not docs:
                    return {
                        "answer": "No relevant information found. Try rephrasing your question.",
                        "total_tokens": 0,
                        "total_cost_usd": 0.0
                    }
                
                # Create context
                context = self._create_context(docs)
                history = self._format_history(chat_history)
                
                # Log conversation context for debugging
                logger.info(f"Processing query: {query[:100]}...")
                logger.info(f"History length: {len(history)} characters")
                logger.info(f"Context length: {len(context)} characters")
                
                # Log conversation context for natural flow
                if history and history != "No previous conversation.":
                    logger.info("Conversation history available - maintaining context")
                
                # Log context quality for citations
                if docs:
                    logger.info(f"Retrieved {len(docs)} documents for context")
                    # Check if documents have page information
                    docs_with_pages = sum(1 for doc in docs if doc.metadata.get('page'))
                    logger.info(f"Documents with page info: {docs_with_pages}/{len(docs)}")
                else:
                    logger.warning("No documents retrieved - may affect citation quality")
                

                
                # Generate response
                prompt = PromptTemplate(
                    template=SYSTEM_TEMPLATE,
                    input_variables=["context", "chat_history", "question"]
                )
                
                prompt_text = prompt.format(
                    context=context,
                    chat_history=history,
                    question=query
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                return {
                    "answer": response.content,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._error_response(f"Processing error: {str(e)}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate document summary."""
        try:
            if not self.vector_manager.is_ready():
                return self._error_response("Vector store not ready")
            
            with get_openai_callback() as cb:
                # Create a retriever that fetches ALL documents to build a complete summary
                doc_count = self.vector_manager.get_document_count()
                retriever = self.vector_manager.vector_store.as_retriever(
                    search_type=Config.SEARCH_TYPE,
                    search_kwargs={"k": min(doc_count, Config.SUMMARY_K)}
                )
                if not retriever:
                    return self._error_response("Could not create retriever")

                # Retrieve a broad sample of all docs
                docs = retriever.invoke("overview")
                # Ensure we don't exceed SUMMARY_K
                if len(docs) > Config.SUMMARY_K:
                    docs = docs[:Config.SUMMARY_K]
                
                if not docs:
                    return self._error_response("No documents for summary")
                
                context = self._create_context(docs)
                
                prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["context"])
                prompt_text = prompt.format(context=context)
                
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                return {
                    "answer": response.content,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._error_response(f"Summary error: {str(e)}")
    
    def _create_context(self, documents: List) -> str:
        """Create context from documents with length control."""
        if not documents:
            return "No documents found."
        
        context_parts = []
        total_chars = 0
        max_chars = getattr(Config, "MAX_CONTEXT_CHARS", 12000)
        
        for doc in documents:
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page')

            # Create source header strictly matching citation format
            if page is not None:
                header = f"[{source}, Page: {page}]"
            else:
                header = f"[{source}]"
            
            part = f"{header}\n{doc.page_content}\n"
            if total_chars + len(part) > max_chars:
                break
            context_parts.append(part)
            total_chars += len(part)
        
        return "\n".join(context_parts)
    
    def _format_history(self, messages: List[Dict]) -> str:
        """Format chat history with proper role handling."""
        if len(messages) <= 1:
            return "No previous conversation."
        
        # Get relevant messages (skip welcome message and current message)
        relevant = []
        for i, msg in enumerate(messages):
            # Skip welcome message and current message (last one)
            if i == 0 or i == len(messages) - 1:
                continue
            if not self._is_welcome_message(msg):
                relevant.append(msg)
        
        # Take last 3 exchanges (6 messages: 3 user + 3 assistant) to understand conversation flow
        if len(relevant) > 6:
            relevant = relevant[-6:]
        
        history_parts = []
        i = 0
        while i < len(relevant):
            # Find user message
            user_msg = None
            assistant_msg = None
            
            # Look for user message
            while i < len(relevant) and relevant[i]["role"] != "user":
                i += 1
            if i < len(relevant):
                user_msg = relevant[i]
                i += 1
            
            # Look for corresponding assistant message
            while i < len(relevant) and relevant[i]["role"] != "assistant":
                i += 1
            if i < len(relevant):
                assistant_msg = relevant[i]
                i += 1
            
            # Add the exchange if we have both messages
            if user_msg and assistant_msg:
                # Truncate content to avoid context overflow but keep enough for context
                user_content = user_msg['content'][:200] + "..." if len(user_msg['content']) > 200 else user_msg['content']
                assistant_content = assistant_msg['content'][:300] + "..." if len(assistant_msg['content']) > 300 else assistant_msg['content']
                
                history_parts.append(f"Human: {user_content}")
                history_parts.append(f"Assistant: {assistant_content}")
        
        history_text = "\n".join(history_parts) if history_parts else "No previous conversation."
        # Truncate history to keep prompt within context window
        max_history_chars = 4000
        if len(history_text) > max_history_chars:
            history_text = history_text[-max_history_chars:]
        return history_text
    
    def _is_welcome_message(self, message: Dict) -> bool:
        """Check if message is welcome message."""
        content = message.get('content', '').lower()
        return 'welcome' in content or 'technical document assistant' in content or 'technical documentation assistant' in content
    

    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "answer": f"Error: {message}. Please try again.",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0
        }