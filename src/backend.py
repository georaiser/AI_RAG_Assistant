# backend.py
"""
RAG Engine for document question-answering and summarization.
Handles query processing, retrieval, and response generation with improved logging.
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
            logger.error("Vector store not available for query processing")
            return self._error_response("Vector store not available")
        
        try:
            logger.info(f"Processing query with {len(chat_history)} history messages")
            
            with get_openai_callback() as cb:
                # Create retrieval chain
                chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_manager.get_retriever(),
                    combine_docs_chain_kwargs={"prompt": self._create_prompt()},
                    return_source_documents=True,
                    verbose=False
                )
                
                # Format chat history (exclude system messages and current query)
                formatted_history = self._format_chat_history(chat_history)
                logger.debug(f"Formatted chat history: {len(formatted_history)} pairs")
                
                # Get response
                result = chain.invoke({
                    "question": query,
                    "chat_history": formatted_history
                })
                
                logger.info(f"Query processed successfully. Tokens used: {cb.total_tokens}")
                
                return {
                    "answer": result["answer"],
                    "source_documents": result.get("source_documents", []),
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
            logger.error("Vector store not available for summary generation")
            return self._error_response("Vector store not available")
        
        try:
            logger.info("Starting document summary generation")
            
            with get_openai_callback() as cb:
                # Retrieve relevant documents for summary
                retriever = self.vector_manager.get_retriever()
                
                # Use a comprehensive query to get diverse content
                summary_query = "comprehensive technical overview documentation methods procedures data results"
                docs = retriever.get_relevant_documents(summary_query)
                
                logger.info(f"Retrieved {len(docs)} documents for summary")
                
                # Create summary prompt
                summary_prompt = PromptTemplate(
                    template=SUMMARY_TEMPLATE,
                    input_variables=["context"]
                )
                
                # Combine document content with better formatting
                context_parts = []
                for doc in docs:
                    source = doc.metadata.get('source', 'Unknown')
                    content_type = doc.metadata.get('content_type', 'text')
                    
                    if content_type == 'table':
                        context_parts.append(f"[TABLE FROM {source}]\n{doc.page_content}\n")
                    else:
                        context_parts.append(f"[DOCUMENT: {source}]\n{doc.page_content}\n")
                
                context = "\n".join(context_parts)
                
                # Generate summary
                prompt_text = summary_prompt.format(context=context)
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                logger.info(f"Summary generated successfully. Tokens used: {cb.total_tokens}")
                
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
    
    def _create_prompt(self) -> PromptTemplate:
        """Create prompt template for question answering."""
        return PromptTemplate(
            template=SYSTEM_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )
    
    def _format_chat_history(self, messages: List[Dict]) -> List[Tuple[str, str]]:
        """Format chat history for LangChain - improved role handling."""
        formatted_history = []
        
        # Filter out system/assistant welcome messages and current user message
        chat_messages = []
        for msg in messages[:-1]:  # Exclude current user message
            if msg.get('role') in ['user', 'assistant'] and not self._is_welcome_message(msg):
                chat_messages.append(msg)
        
        # Pair user and assistant messages
        for i in range(0, len(chat_messages), 2):
            if i + 1 < len(chat_messages):
                user_msg = chat_messages[i]
                assistant_msg = chat_messages[i + 1]
                
                # Ensure proper pairing
                if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                    formatted_history.append((
                        user_msg['content'],
                        assistant_msg['content']
                    ))
        
        logger.debug(f"Formatted {len(formatted_history)} chat history pairs")
        return formatted_history
    
    def _is_welcome_message(self, message: Dict) -> bool:
        """Check if message is a welcome message."""
        content = message.get('content', '').lower()
        return 'welcome' in content and 'technical document assistant' in content
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response with default values."""
        error_response = {
            "answer": f"I apologize, but I encountered an error: {message}. Please try again.",
            "source_documents": [],
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0.0
        }
        logger.warning(f"Returning error response: {message}")
        return error_response


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


# Convenience functions for backward compatibility
def handle_query(query: str, messages: List[Dict]) -> Dict[str, Any]:
    """Handle user query with RAG engine."""
    engine = RAGEngine()
    return engine.process_query(query, messages)