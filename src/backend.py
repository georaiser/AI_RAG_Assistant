# backend.py
"""
Unified RAG Engine with simplified role handling - only user/assistant roles.
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

# UNIFIED ROLE CONSTANTS - Only two roles needed
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


class RAGEngine:
    """Unified RAG engine with simplified role handling."""
    
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
        """Process a user query with simplified role handling."""
        try:
            chain = self._get_chain()
            if not chain:
                return self._error_response("Failed to create conversation chain")
            
            # Format chat history - simplified, no complex role mapping
            formatted_history = self._format_chat_history(chat_history)
            
            if formatted_history:
                logger.info(f"Using {len(formatted_history)} conversation exchanges for context")
            
            with get_openai_callback() as cb:
                result = chain.invoke({
                    "question": query,
                    "chat_history": formatted_history
                })
                
                # Log retrieval quality
                source_docs = result.get("source_documents", [])
                if source_docs:
                    logger.info(f"Retrieved {len(source_docs)} documents for context")
                    # Check citation format
                    docs_with_proper_headers = sum(
                        1 for doc in source_docs 
                        if doc.page_content.startswith('[') and ']' in doc.page_content
                    )
                    logger.info(f"Documents with proper headers: {docs_with_proper_headers}/{len(source_docs)}")
                else:
                    logger.warning("No source documents retrieved")
                
                # Validate answer completeness
                answer = result["answer"]
                if len(answer.strip()) < 50:
                    logger.warning(f"Short answer generated: {len(answer)} characters")
                
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
        Simplified chat history formatting - only handles user/assistant roles.
        Returns list of (human_message, ai_message) tuples.
        """
        if len(messages) <= 1:
            return []
        
        # Filter out welcome message and current query
        relevant_messages = []
        for i, msg in enumerate(messages):
            # Skip welcome message (first message)
            if i == 0 and self._is_welcome_message(msg):
                continue
            # Skip current query (last message)
            if i == len(messages) - 1:
                continue
            relevant_messages.append(msg)
        
        # Limit to last 6 messages (3 user-assistant pairs)
        if len(relevant_messages) > 6:
            relevant_messages = relevant_messages[-6:]
        
        # Create (user, assistant) pairs - simplified logic
        formatted_history = []
        i = 0
        while i < len(relevant_messages) - 1:
            current_msg = relevant_messages[i]
            next_msg = relevant_messages[i + 1]
            
            # Look for user-assistant pair
            if (current_msg["role"] == USER_ROLE and 
                next_msg["role"] == ASSISTANT_ROLE):
                
                user_content = current_msg["content"]
                assistant_content = next_msg["content"]
                
                # Truncate to prevent context overflow
                user_truncated = user_content[:300] + "..." if len(user_content) > 300 else user_content
                assistant_truncated = assistant_content[:400] + "..." if len(assistant_content) > 400 else assistant_content
                
                formatted_history.append((user_truncated, assistant_truncated))
                i += 2  # Skip both messages
            else:
                i += 1  # Move to next message
        
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
        """Create context from documents with proper citation format."""
        if not documents:
            return "No documents found."
        
        context_parts = []
        total_chars = 0
        max_chars = getattr(Config, "MAX_CONTEXT_CHARS", 12000)
        
        for doc in documents:
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page')

            content = doc.page_content.strip()
            
            # Ensure proper citation header format
            if page is not None:
                expected_header = f"[{source}, Page: {page}]"
            else:
                expected_header = f"[{source}]"
            
            # Fix header format if needed
            lines = content.split('\n')
            first_line = lines[0] if lines else ""
            
            if (not first_line.startswith('[') or 
                '[PAGE ' in first_line.upper() or 
                source not in first_line):
                
                # Remove bad header and add correct one
                if first_line.startswith('[') and ']' in first_line:
                    content_without_header = '\n'.join(lines[1:])
                else:
                    content_without_header = content
                
                content = f"{expected_header}\n{content_without_header}"
            
            # Add to context if it fits
            if total_chars + len(content) + 2 > max_chars:
                break
            
            context_parts.append(content)
            total_chars += len(content) + 2
        
        return "\n\n".join(context_parts)
    
    def _is_welcome_message(self, message: Dict) -> bool:
        """Check if message is a welcome message."""
        content = message.get('content', '').lower()
        welcome_indicators = [
            'welcome', 
            'technical document assistant',
            'capabilities:',
            'what would you like to explore'
        ]
        return any(indicator in content for indicator in welcome_indicators)
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "answer": f"I apologize, but I encountered an error: {message}. Please try rephrasing your question.",
            "source_documents": [],
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0
        }