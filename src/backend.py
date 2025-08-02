# backend.py
"""
RAG Engine for document question-answering and summarization.
Handles query processing, retrieval, and response generation.
"""

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
    
    def process_query(self, query: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Process user query and return response with metadata."""
        if not self.vector_store:
            return self._error_response("Vector store not available")
        
        try:
            with get_openai_callback() as cb:
                # Create retrieval chain
                chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_manager.get_retriever(),
                    combine_docs_chain_kwargs={"prompt": self._create_prompt()},
                    return_source_documents=True,
                    verbose=False
                )
                
                # Format chat history
                formatted_history = self._format_chat_history(chat_history)
                
                # Get response
                result = chain.invoke({
                    "question": query,
                    "chat_history": formatted_history
                })
                
                return {
                    "answer": result["answer"],
                    "source_documents": result.get("source_documents", []),
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            print(f"ERROR processing query: {e}")
            return self._error_response("Failed to process query")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all documents."""
        if not self.vector_store:
            return self._error_response("Vector store not available")
        
        try:
            with get_openai_callback() as cb:
                # Retrieve all documents for summary
                retriever = self.vector_manager.get_retriever()
                docs = retriever.get_relevant_documents("comprehensive document summary")
                
                # Create summary prompt
                summary_prompt = PromptTemplate(
                    template=SUMMARY_TEMPLATE,
                    input_variables=["context"]
                )
                
                # Combine document content
                context = "\n\n".join([
                    f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                    for doc in docs
                ])
                
                # Generate summary
                prompt_text = summary_prompt.format(context=context)
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                
                return {
                    "answer": response.content,
                    "source_documents": docs,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost_usd": cb.total_cost
                }
                
        except Exception as e:
            print(f"ERROR generating summary: {e}")
            return self._error_response("Failed to generate summary")
    
    def _create_prompt(self) -> PromptTemplate:
        """Create prompt template for question answering."""
        return PromptTemplate(
            template=SYSTEM_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )
    
    def _format_chat_history(self, messages: List[Dict]) -> List[Tuple[str, str]]:
        """Format chat history for LangChain."""
        formatted_history = []
        
        # Skip welcome message and last user message
        chat_history_messages = messages[1:-1]
        
        for i in range(0, len(chat_history_messages), 2):
            if i + 1 < len(chat_history_messages):
                user_msg = chat_history_messages[i]
                bot_msg = chat_history_messages[i + 1]
                if user_msg.get('role') == 'user' and bot_msg.get('role') == 'bot':
                    formatted_history.append((
                        user_msg['content'],
                        bot_msg['content']
                    ))
        
        return formatted_history
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response with default values."""
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
            
            print(f"Updated embedding model: {model_type} - {model_name}")
            return True
            
        except Exception as e:
            print(f"ERROR updating embedding model: {e}")
            return False


# Convenience functions for backward compatibility
def handle_query(query: str, messages: List[Dict]) -> Dict[str, Any]:
    """Handle user query with RAG engine."""
    engine = RAGEngine()
    return engine.process_query(query, messages)