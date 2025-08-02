# backend.py
"""
Improved RAG Engine with better citation handling and source verification.
"""

import logging
from typing import Dict, List, Tuple, Any
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from vector_store import VectorStoreManager
from prompts import SYSTEM_TEMPLATE, SUMMARY_TEMPLATE
from config import Config
from pydantic import SecretStr

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main RAG engine for document question-answering."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=SecretStr(Config.OPENAI_API_KEY)
        )
        self.vector_manager = VectorStoreManager()
        self.vector_store = self.vector_manager.load_vector_store()
        logger.info(f"RAG Engine initialized with model: {Config.MODEL_NAME}")
    
    def process_query(self, query: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Process user query and return response with metadata."""
        if not self.vector_store:
            logger.error("Vector store not available")
            return self._error_response("Vector store not available")
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            with get_openai_callback() as cb:
                # Get relevant documents first
                retriever = self.vector_manager.get_retriever()
                source_docs = retriever.get_relevant_documents(query)
                
                # Create enhanced context with source information
                enhanced_context = self._create_enhanced_context(source_docs)
                
                # Format chat history
                formatted_history = self._format_chat_history(chat_history)
                
                # Create prompt with enhanced context
                prompt = PromptTemplate(
                    template=SYSTEM_TEMPLATE,
                    input_variables=["context", "chat_history", "question"]
                )
                
                # Generate response
                prompt_text = prompt.format(
                    context=enhanced_context,
                    chat_history=self._format_history_for_prompt(formatted_history),
                    question=query
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                logger.info(f"Query processed. Tokens: {cb.total_tokens}")
                
                return {
                    "answer": response.content,
                    "source_documents": source_docs,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._error_response("Failed to process query")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all documents."""
        if not self.vector_store:
            logger.error("Vector store not available")
            return self._error_response("Vector store not available")
        
        try:
            logger.info("Generating document summary")
            
            with get_openai_callback() as cb:
                retriever = self.vector_manager.get_retriever()
                
                # Get diverse content for summary
                summary_query = "overview technical methods procedures data results implementation"
                docs = retriever.get_relevant_documents(summary_query)
                
                logger.info(f"Retrieved {len(docs)} documents for summary")
                
                # Create enhanced context
                enhanced_context = self._create_enhanced_context(docs)
                
                # Generate summary
                prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["context"])
                prompt_text = prompt.format(context=enhanced_context)
                
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                logger.info(f"Summary generated. Tokens: {cb.total_tokens}")
                
                return {
                    "answer": response.content,
                    "source_documents": docs,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._error_response("Failed to generate summary")
    
    def _create_enhanced_context(self, documents: List) -> str:
        """Create enhanced context with proper source information."""
        context_parts = []
        
        for doc in documents:
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            content_type = metadata.get('content_type', 'text')
            
            # Create source header
            if content_type == 'table':
                if 'page' in metadata:
                    header = f"[TABLE - Source: {source}, Page: {metadata['page']}]"
                else:
                    header = f"[TABLE - Source: {source}]"
            else:
                if 'page' in metadata:
                    header = f"[DOCUMENT - Source: {source}, Page: {metadata['page']}]"
                else:
                    header = f"[DOCUMENT - Source: {source}]"
            
            context_parts.append(f"{header}\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _format_chat_history(self, messages: List[Dict]) -> List[Tuple[str, str]]:
        """Format chat history for conversation chain."""
        formatted_history = []
        
        # Filter and pair messages
        chat_messages = [msg for msg in messages[:-1] if not self._is_welcome_message(msg)]
        
        for i in range(0, len(chat_messages), 2):
            if i + 1 < len(chat_messages):
                user_msg = chat_messages[i]
                assistant_msg = chat_messages[i + 1]
                
                if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                    formatted_history.append((user_msg['content'], assistant_msg['content']))
        
        return formatted_history
    
    def _format_history_for_prompt(self, history: List[Tuple[str, str]]) -> str:
        """Format history for prompt template."""
        if not history:
            return "No previous conversation."
        
        formatted = []
        for user_msg, assistant_msg in history[-3:]:  # Keep last 3 exchanges
            formatted.append(f"User: {user_msg}")
            formatted.append(f"Assistant: {assistant_msg}")
        
        return "\n".join(formatted)
    
    def _is_welcome_message(self, message: Dict) -> bool:
        """Check if message is a welcome message."""
        content = message.get('content', '').lower()
        return 'welcome' in content or 'technical document assistant' in content
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "answer": f"I apologize, but I encountered an error: {message}. Please try again.",
            "source_documents": [],
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0
        }


class EmbeddingManager:
    """Manages embedding model selection and configuration."""
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get available embedding models."""
        return {
            "OpenAI": Config.AVAILABLE_OPENAI_EMBEDDINGS,
            "HuggingFace": Config.AVAILABLE_HF_EMBEDDINGS
        }
    
    @staticmethod
    def update_embedding_model(model_type: str, model_name: str) -> bool:
        """Update embedding model configuration."""
        try:
            if model_type.lower() == "openai":
                Config.EMBEDDING_TYPE = "openai"
                Config.OPENAI_EMBEDDING_MODEL = model_name
            else:
                Config.EMBEDDING_TYPE = "huggingface"
                Config.HF_EMBEDDING_MODEL = model_name
            
            logger.info(f"Updated embedding model: {model_type} - {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating embedding model: {e}")
            return False