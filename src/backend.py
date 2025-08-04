# backend.py
"""
RAG Engine with fixed role handling and improved citations.
"""

import logging
from typing import Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage
from vector_store import VectorStoreManager
from prompts import SYSTEM_TEMPLATE, SUMMARY_TEMPLATE
from config import Config
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# Simple role constants
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


class RAGEngine:
    """RAG engine with fixed conversation handling."""
    
    def __init__(self, vector_manager: VectorStoreManager):
        """Initialize RAG engine."""
        self.vector_manager = vector_manager
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=SecretStr(Config.OPENAI_API_KEY)
        )
        self._retriever = None
        self._chain = None
        logger.info(f"RAG Engine initialized with {Config.MODEL_NAME}")
    
    def _get_retriever(self):
        """Get or create retriever (cached)."""
        if self._retriever is None:
            self._retriever = self.vector_manager.get_retriever()
        return self._retriever
    
    def _get_chain(self):
        """Get or create conversational chain (cached)."""
        if self._chain is None:
            retriever = self._get_retriever()
            if not retriever:
                return None
            
            prompt = PromptTemplate(
                template=SYSTEM_TEMPLATE,
                input_variables=["context", "chat_history", "question"]
            )
            
            self._chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=Config.DEBUG_MODE
            )
        
        return self._chain
    
    def process_query(self, query: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Process a user query with improved error handling."""
        try:
            chain = self._get_chain()
            if not chain:
                return self._error_response("Failed to create conversation chain")
            
            # Format chat history - exclude current query and welcome message
            formatted_history = self._format_chat_history(chat_history)
            
            if Config.DEBUG_MODE:
                logger.info(f"Processing query: {query[:100]}...")
                logger.info(f"Chat history length: {len(formatted_history)} pairs")
            
            with get_openai_callback() as cb:
                result = chain.invoke({
                    "question": query,
                    "chat_history": formatted_history
                })
                
                answer = result.get("answer", "")
                source_docs = result.get("source_documents", [])
                
                # Log retrieval info
                if Config.DEBUG_MODE:
                    logger.info(f"Retrieved {len(source_docs)} documents")
                    logger.info(f"Answer length: {len(answer)} characters")
                    if Config.VERBOSE_MODE:
                        for i, doc in enumerate(source_docs[:3]):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            logger.info(f"Source {i+1}: {source}, Page: {page}")
                
                return {
                    "answer": answer,
                    "source_documents": source_docs,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._error_response(f"Processing error: {str(e)}")
    
    def _format_chat_history(self, messages: List[Dict]) -> List[Tuple[str, str]]:
        """
        Fixed chat history formatting.
        Returns list of (human_message, ai_message) tuples.
        """
        if len(messages) <= 1:
            return []
        
        # Filter messages - exclude welcome message and current query
        filtered_messages = []
        for i, msg in enumerate(messages):
            # Skip welcome message (first message with "Welcome")
            if i == 0 and "Welcome" in msg.get("content", ""):
                continue
            # Skip current query (last message)
            if i == len(messages) - 1:
                continue
            filtered_messages.append(msg)
        
        if not filtered_messages:
            return []
        
        # Keep only last 6 messages (3 exchanges) to prevent context overflow
        if len(filtered_messages) > 6:
            filtered_messages = filtered_messages[-6:]
        
        # Create user-assistant pairs
        formatted_history = []
        i = 0
        while i < len(filtered_messages) - 1:
            current_msg = filtered_messages[i]
            next_msg = filtered_messages[i + 1]
            
            if (current_msg["role"] == USER_ROLE and 
                next_msg["role"] == ASSISTANT_ROLE):
                
                # Truncate long messages to prevent context overflow
                user_content = current_msg["content"][:400]
                assistant_content = next_msg["content"][:500]
                
                formatted_history.append((user_content, assistant_content))
                i += 2
            else:
                i += 1
        
        if Config.VERBOSE_MODE and formatted_history:
            logger.info(f"Formatted {len(formatted_history)} conversation pairs")
        
        return formatted_history
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate document summary."""
        try:
            if not self.vector_manager.is_ready():
                return self._error_response("Vector store not ready")
            
            with get_openai_callback() as cb:
                retriever = self._get_retriever()
                if not retriever:
                    return self._error_response("Could not create retriever")

                # Retrieve documents for summary
                docs = retriever.invoke("overview summary content")
                
                if len(docs) > Config.SUMMARY_K:
                    docs = docs[:Config.SUMMARY_K]
                
                if not docs:
                    return self._error_response("No documents retrieved for summary")
                
                context = self._create_context(docs)
                
                if Config.DEBUG_MODE:
                    logger.info(f"Creating summary from {len(docs)} documents")
                    logger.info(f"Context length: {len(context)} characters")
                
                prompt = PromptTemplate(
                    template=SUMMARY_TEMPLATE, 
                    input_variables=["context"]
                )
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
        """Create context from documents with proper citations."""
        if not documents:
            return "No documents found."
        
        # Group documents by source file
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        context_parts = []
        total_chars = 0
        max_chars = Config.MAX_CONTEXT_CHARS
        
        # Process each source file
        for source, docs in docs_by_source.items():
            if total_chars >= max_chars:
                break
                
            # Sort documents by page number if available
            docs.sort(key=lambda d: d.metadata.get('page', 0))
            
            # Process each document
            for doc in docs:
                if total_chars >= max_chars:
                    break
                
                content = doc.page_content.strip()
                if not content:
                    continue
                
                # Ensure proper citation header
                content = self._ensure_citation_header(content, doc.metadata)
                
                # Check if adding this would exceed limit
                if total_chars + len(content) + 2 > max_chars:
                    break
                
                context_parts.append(content)
                total_chars += len(content) + 2
        
        final_context = "\n\n".join(context_parts)
        
        if Config.DEBUG_MODE:
            logger.info(f"Created context with {len(context_parts)} parts, {total_chars} chars")
        
        return final_context
    
    def _ensure_citation_header(self, content: str, metadata: Dict) -> str:
        """Ensure content has proper citation header."""
        lines = content.split('\n')
        first_line = lines[0] if lines else ""
        
        # Get source info
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page')
        
        # Create proper header
        if page is not None:
            correct_header = f"[{source}, Page: {page}]"
        else:
            correct_header = f"[{source}]"
        
        # Check if first line is already a proper citation
        if first_line.startswith('[') and ']' in first_line and source in first_line:
            # Update existing header to ensure page is included if available
            if page is not None and f"Page: {page}" not in first_line:
                content_body = '\n'.join(lines[1:]).strip()
                return f"{correct_header}\n{content_body}"
            return content
        else:
            # Remove any existing bad header and add correct one
            if first_line.startswith('[') and ']' in first_line:
                content_body = '\n'.join(lines[1:]).strip()
            else:
                content_body = content.strip()
            
            return f"{correct_header}\n{content_body}"
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        error_msg = f"I apologize, but I encountered an error: {message}. Please try rephrasing your question."
        
        if Config.DEBUG_MODE:
            logger.error(f"Error response: {message}")
        
        return {
            "answer": error_msg,
            "source_documents": [],
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0
        }